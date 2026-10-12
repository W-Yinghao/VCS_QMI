"""r7 R7-30 D1-20 — three same-architecture selectors trained on the D1-10 FIT tuples (frozen protocol D1_20; OPEN-5 / OPEN-6 details in
reports/R7/R7_D1_20_OPEN5_DRAFT.yaml, frozen after the D1-10 CAL readout and before any D1-20 run).
  backbone  : 2-layer pre-LN transformer (d 512, 8 heads, FFN 2048, dropout 0.1) over [question tokens (frozen LLaVA input-embedding lookup,
              4096 -> 512) ; 576 visual tokens (cached LLaVA projector output, PCA 4096 -> 512 fitted on FIT, then 512 -> 512, learned 24 × 24
              position embedding)]; keys are masked with an additive per-key bias (0 keep / −1e9 drop / log m soft), so a dropped token
              influences nothing
  critic    : attention pooling -> MLP -> f, T = tanh f;  T_B on all tokens, T_A on the kept tokens (the compressed observation (q, V_K, K))
  objective : S1 'vcs' maximises the original J of T_B and T_A (P: C = +1, Q: C = −1, class weights), r(K) = (T_B − T_A(K))²;
              S2 'logistic' balanced logistic on 2f for T_B and T_A, same read-out;  S3 'direct' regresses the teacher's realised
              h = (C − a)² − (C − b)² from [pool_B, pool_A] (class-weighted MSE), r(K) = ĥ
  phase A   : estimator on the D1-10 masks (64 questions × 4 masks per step; AdamW, 1,000-step linear warm-up then cosine); every
              `eval_every` steps CAL Spearman(r(K), realised h) over all
              CAL (question, mask) pairs; the best state is kept (same rule for S1–S3)
  phase B   : gate MLP on the detached full-pass visual outputs; per question a budget k from the grid; relaxed top-k
              m = sigmoid((g + Gumbel − t_k) / tau), t_k the k-th largest perturbed logit (detached), tau annealed 1 -> 0.1 (cosine);
              the frozen estimator's predicted risk of the soft-masked observation is minimised; deployment = hard top-k of g;
              soft / hard gap on CAL recorded
  dynamic   : tau_dyn on CAL so the mean kept fraction hits 25 / 50 / 75 % (smallest budget with r(top-k) <= tau_dyn)
Torch only (any env / GPU type).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9


@dataclass
class Cfg:
    emb_dim: int = 4096
    vis_dim: int = 512
    n_vis: int = 576
    grid: int = 24
    d: int = 512
    heads: int = 8
    layers: int = 2
    ffn: int = 2048
    dropout: float = 0.1
    budgets: tuple = (72, 144, 288, 432)
    steps_a: int = 20000
    steps_b: int = 5000
    q_per_step: int = 64
    masks_per_q: int = 4
    lr: float = 3e-4
    lr_gate: float = 3e-4
    wd: float = 0.01
    eval_every: int = 1000
    warmup: int = 1000                             # coordinator decision 2026-10-12: linear warm-up then cosine, every objective / ablation
    tau_start: float = 1.0
    tau_end: float = 0.1
    seed: int = 0
    no_question: bool = False                     # ablation A1
    target: str = "h"                             # 'h' or 'd' (ablation A2, direct objective only)
    extra: dict = field(default_factory=dict)


class Block(nn.Module):
    def __init__(self, d, h, ffn, p):
        super().__init__()
        self.h, self.p = h, p; self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d); self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d)); self.drop = nn.Dropout(p)

    def forward(self, x, bias):                                                        # bias [B, 1, 1, L], additive per key
        B, L, D = x.shape; q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class Selector(nn.Module):
    def __init__(self, cfg: Cfg, objective: str):
        super().__init__()
        assert objective in ("vcs", "logistic", "direct"); self.cfg, self.objective = cfg, objective; d = cfg.d
        self.q_proj, self.v_proj = nn.Linear(cfg.emb_dim, d), nn.Linear(cfg.vis_dim, d)
        self.pos = nn.Parameter(torch.randn(cfg.n_vis, d) * 0.02); self.typ = nn.Parameter(torch.randn(2, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, cfg.ffn, cfg.dropout) for _ in range(cfg.layers)]); self.ln = nn.LayerNorm(d)
        self.pool_q = nn.Parameter(torch.randn(d) * 0.02)
        self.critic = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.risk = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.gate = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def estimator_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("gate.")]

    def encode(self, q_emb, q_pad, v, vis_bias):
        """q_emb [B, Lq, E] (q_pad True = padding), v [B, N, Vd], vis_bias [B, N] additive -> token outputs [B, Lq + N, d], pooled [B, d]."""
        B = v.shape[0]; xq = self.q_proj(q_emb) + self.typ[0]; xv = self.v_proj(v) + self.pos + self.typ[1]
        qb = torch.where(q_pad, torch.full_like(q_pad, NEG, dtype=xv.dtype), torch.zeros_like(q_pad, dtype=xv.dtype))
        if self.cfg.no_question:                                                       # ablation A1: question tokens invisible to every key
            qb = torch.full_like(qb, NEG)
        x = torch.cat([xq, xv], 1); kb = torch.cat([qb, vis_bias.to(xv.dtype)], 1)
        for blk in self.blocks:
            x = blk(x, kb[:, None, None, :])
        x = self.ln(x); w = torch.softmax((x @ self.pool_q) / math.sqrt(x.shape[-1]) + kb, 1)
        return x, (w[..., None] * x).sum(1)

    def T(self, pooled):
        f = self.critic(pooled).squeeze(-1)
        return torch.tanh(f), f

    def gates(self, tokens):
        return self.gate(tokens[:, -self.cfg.n_vis:].detach()).squeeze(-1)

    def readout(self, pooled_b, pooled_a):
        if self.objective == "direct":
            return self.risk(torch.cat([pooled_b, pooled_a], -1)).squeeze(-1)
        return (self.T(pooled_b)[0] - self.T(pooled_a)[0]) ** 2


def hard_bias(keep):
    """keep [B, N] bool -> additive key bias."""
    return torch.where(keep, torch.zeros((), device=keep.device), torch.full((), NEG, device=keep.device))


def a_plus(t):
    return t - 0.5 * t * t


def a_minus(t):
    return -t - 0.5 * t * t


def estimator_loss(model, batch, wts):
    """batch: q_emb, q_pad, v (per question); keep [Q, M, N]; C [Q]; h, d [Q, M].  Returns the objective's loss."""
    Q, M, N = batch["keep"].shape; vb = torch.zeros(Q, N, device=batch["v"].device)
    _, pb = model.encode(batch["q_emb"], batch["q_pad"], batch["v"], vb)
    rep = lambda x: x.repeat_interleave(M, 0)  # noqa: E731
    _, pa = model.encode(rep(batch["q_emb"]), rep(batch["q_pad"]), rep(batch["v"]), hard_bias(batch["keep"].reshape(Q * M, N)))
    C = batch["C"].float(); w = torch.where(C > 0, wts[0], wts[1]); Cm, wm = rep(C), rep(w)
    if model.objective == "direct":
        y = (batch["h"] if model.cfg.target == "h" else batch["d"]).reshape(-1)
        return (wm * (model.readout(rep(pb), pa) - y) ** 2).sum() / wm.sum()
    tb, fb = model.T(pb); ta, fa = model.T(pa)
    if model.objective == "vcs":
        jb = (w * torch.where(C > 0, a_plus(tb), a_minus(tb))).sum() / w.sum()
        ja = (wm * torch.where(Cm > 0, a_plus(ta), a_minus(ta))).sum() / wm.sum()
        return -(jb + ja)
    yb, ya = (C > 0).float(), (Cm > 0).float()
    lb = (w * F.binary_cross_entropy_with_logits(2 * fb, yb, reduction="none")).sum() / w.sum()
    la = (wm * F.binary_cross_entropy_with_logits(2 * fa, ya, reduction="none")).sum() / wm.sum()
    return lb + la


def relaxed_topk(g, k, tau, gen=None):
    """Soft keep mask: sigmoid((g + Gumbel − t_k) / tau) with t_k midway between the k-th and (k+1)-th largest perturbed logits (detached),
    so the mask sums to k as tau -> 0 (for k = N the threshold sits one unit below the smallest); k [B] ints."""
    u = torch.rand(g.shape, generator=gen, device=g.device).clamp_(1e-9, 1 - 1e-9); z = g - torch.log(-torch.log(u))
    zs = torch.sort(z.detach(), 1, descending=True).values; N = g.shape[1]
    hi = zs.gather(1, (k - 1).clamp(0, N - 1)[:, None]); lo = zs.gather(1, k.clamp(max=N - 1)[:, None])
    t = torch.where((k >= N)[:, None], hi - 1.0, 0.5 * (hi + lo))
    return torch.sigmoid((z - t) / tau)


def hard_topk(g, k):
    """Deployment mask: exactly k[b] highest gate logits per row."""
    idx = torch.argsort(g, 1, descending=True); keep = torch.zeros_like(g, dtype=torch.bool)
    ranks = torch.empty_like(idx); ranks.scatter_(1, idx, torch.arange(g.shape[1], device=g.device).expand_as(idx))
    return ranks < k[:, None]


def gate_loss(model, batch, k, tau, gen=None):
    Q, N = batch["v"].shape[0], model.cfg.n_vis; vb = torch.zeros(Q, N, device=batch["v"].device)
    tok, pb = model.encode(batch["q_emb"], batch["q_pad"], batch["v"], vb)
    m = relaxed_topk(model.gates(tok), k, tau, gen)
    _, pa = model.encode(batch["q_emb"], batch["q_pad"], batch["v"], torch.log(m + 1e-6))
    return model.readout(pb, pa).mean()


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a, kind="stable"), kind="stable"), np.argsort(np.argsort(b, kind="stable"), kind="stable")
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) > 0 and np.std(rb) > 0 else float("nan")


@torch.no_grad()
def predict_r(model, data, idx, chunk=64):
    """Predicted risk r(K) for every (question, mask) pair of the questions `idx` -> [len(idx), M]."""
    model.eval(); out = []
    for c in range(0, len(idx), chunk):
        b = data.batch(idx[c:c + chunk], all_masks=True); Q, M, N = b["keep"].shape
        _, pb = model.encode(b["q_emb"], b["q_pad"], b["v"], torch.zeros(Q, N, device=b["v"].device))
        rep = lambda x: x.repeat_interleave(M, 0)  # noqa: E731
        _, pa = model.encode(rep(b["q_emb"]), rep(b["q_pad"]), rep(b["v"]), hard_bias(b["keep"].reshape(Q * M, N)))
        out.append(model.readout(rep(pb), pa).reshape(Q, M).float().cpu())
    model.train()
    return torch.cat(out).numpy()


class TupleData:
    """D1-10 teacher records + cached features on one device.  feats [n_img, N, vis_dim]; img_of [n_q]; q_ids [n_q, Lq] (0 = pad);
    emb_table [V, E]; keep [n_q, M, N] bool; C [n_q]; h, d [n_q, M] on the ±1 scale."""

    def __init__(self, feats, img_of, q_ids, q_len, emb_table, keep, C, h, d, device):
        self.dev = device; self.feats, self.img_of = feats.to(device), torch.as_tensor(img_of, device=device)
        self.q_ids, self.q_len, self.emb = torch.as_tensor(q_ids, device=device), torch.as_tensor(q_len, device=device), emb_table.to(device)
        self.keep, self.C = torch.as_tensor(keep, device=device), torch.as_tensor(C, device=device)
        self.h, self.d = torch.as_tensor(h, dtype=torch.float32, device=device), torch.as_tensor(d, dtype=torch.float32, device=device)
        self.n = len(C)

    def batch(self, qi, mi=None, all_masks=False):
        qi = torch.as_tensor(qi, device=self.dev); ids = self.q_ids[qi]; L = int(self.q_len[qi].max())
        ids = ids[:, :L]; pad = torch.arange(L, device=self.dev)[None, :] >= self.q_len[qi][:, None]
        keep = self.keep[qi] if all_masks else self.keep[qi[:, None], torch.as_tensor(mi, device=self.dev)]
        h = self.h[qi] if all_masks else self.h[qi[:, None], torch.as_tensor(mi, device=self.dev)]
        d = self.d[qi] if all_masks else self.d[qi[:, None], torch.as_tensor(mi, device=self.dev)]
        return {"q_emb": self.emb[ids].float(), "q_pad": pad, "v": self.feats[self.img_of[qi]].float(), "keep": keep, "C": self.C[qi], "h": h, "d": d}


def cosine_lr(step, total, base, warmup=0):
    """Linear warm-up over `warmup` steps, then cosine from `base` to 0 over the remaining steps."""
    if warmup and step <= warmup:
        return base * step / warmup
    return base * 0.5 * (1 + math.cos(math.pi * min(step - warmup, total - warmup) / max(total - warmup, 1)))


def _rng_state(rng, gen):
    st = {"np": rng.bit_generator.state, "torch": torch.get_rng_state(), "gen": gen.get_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st, rng, gen):
    rng.bit_generator.state = st["np"]; torch.set_rng_state(st["torch"]); gen.set_state(st["gen"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def train(cfg: Cfg, objective: str, fit: TupleData, cal: TupleData, wts, log_path: Path | None = None, ckpt_path: Path | None = None,
          resume_path: Path | None = None, stop_after: int | None = None):
    """Phase A (estimator, CAL-selected) then phase B (gates, fixed steps).  With resume_path, state is saved at every evaluation and at the
    end of phase A; a restarted unit continues bit-for-bit from the last save (stop_after simulates an interruption).  Returns (model, log)."""
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); dev = fit.dev; gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    model = Selector(cfg, objective).to(dev); wts = torch.as_tensor(wts, dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(model.estimator_parameters(), lr=cfg.lr, weight_decay=cfg.wd); rng = np.random.default_rng(cfg.seed)
    M = fit.keep.shape[1]; cal_idx = np.arange(cal.n); cal_h = cal.h.cpu().numpy(); log = {"cfg": asdict(cfg), "objective": objective, "evals": []}
    best = (-np.inf, None, 0); t0 = time.time(); start, phase_a_done = 1, False
    if resume_path is not None and Path(resume_path).exists():
        ck = torch.load(resume_path, map_location=dev, weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        best, log, start, phase_a_done = ck["best"], ck["log"], ck["step"] + 1, ck.get("phase_a_done", False); _set_rng_state(ck["rng"], rng, gen)
        log.setdefault("resumed_at", []).append(ck["step"])

    def save(step, done=False):
        if resume_path is not None:
            tmp = Path(str(resume_path) + ".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "best": best, "log": log, "step": step, "phase_a_done": done,
                        "rng": _rng_state(rng, gen)}, tmp)
            tmp.replace(resume_path)
    if not phase_a_done:
        for step in range(start, cfg.steps_a + 1):
            for gpar in opt.param_groups:
                gpar["lr"] = cosine_lr(step, cfg.steps_a, cfg.lr, min(cfg.warmup, cfg.steps_a))
            qi = rng.choice(fit.n, size=min(cfg.q_per_step, fit.n), replace=False); mi = np.stack([rng.choice(M, size=cfg.masks_per_q, replace=False) for _ in qi])
            loss = estimator_loss(model, fit.batch(qi, mi), wts); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step % cfg.eval_every == 0 or step == cfg.steps_a:
                r = predict_r(model, cal, cal_idx); rho = spearman(r.reshape(-1), cal_h.reshape(-1))
                log["evals"].append({"step": step, "loss": float(loss), "cal_spearman_r_h": rho, "seconds": time.time() - t0})
                if np.isfinite(rho) and rho > best[0]:
                    best = (rho, {k: v.detach().clone() for k, v in model.state_dict().items()}, step)
                save(step)
                if stop_after is not None and step >= stop_after:
                    return model, log
        if best[1] is not None:
            model.load_state_dict(best[1])
        log["selected_step"], log["selected_cal_spearman"] = best[2], best[0]; save(cfg.steps_a, done=True)
    for p in model.estimator_parameters():                                             # phase B: estimator frozen
        p.requires_grad_(False)
    gopt = torch.optim.AdamW(model.gate.parameters(), lr=cfg.lr_gate, weight_decay=cfg.wd); budgets = torch.as_tensor(cfg.budgets, device=dev)
    gl = None
    for step in range(1, cfg.steps_b + 1):
        tau = cfg.tau_end + 0.5 * (cfg.tau_start - cfg.tau_end) * (1 + math.cos(math.pi * step / max(cfg.steps_b, 1)))
        qi = rng.choice(fit.n, size=min(cfg.q_per_step, fit.n), replace=False); b = fit.batch(qi, np.zeros((len(qi), 1), dtype=int))
        k = budgets[torch.randint(len(cfg.budgets), (len(qi),), generator=gen, device=dev)]
        gl = gate_loss(model, b, k, tau, gen); gopt.zero_grad(set_to_none=True); gl.backward(); gopt.step()
    log["gate_final_loss"] = float(gl) if gl is not None else None
    log["soft_hard_gap"] = soft_hard_gap(model, cal, cfg, gen) if cfg.steps_b else None
    log["seconds"] = time.time() - t0
    if log_path:
        Path(log_path).write_text(json.dumps(log, indent=1))
    if ckpt_path:
        torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg), "objective": objective}, ckpt_path)
    return model, log


@torch.no_grad()
def soft_hard_gap(model, data, cfg, gen=None, chunk=64):
    """Per budget: mean predicted risk of the relaxed mask at the final temperature vs the deployed hard top-k mask (CAL)."""
    model.eval(); res = {}
    for kk in cfg.budgets:
        soft, hard = [], []
        for c in range(0, data.n, chunk):
            qi = np.arange(c, min(c + chunk, data.n)); b = data.batch(qi, np.zeros((len(qi), 1), dtype=int)); Q, N = len(qi), cfg.n_vis
            tok, pb = model.encode(b["q_emb"], b["q_pad"], b["v"], torch.zeros(Q, N, device=data.dev)); g = model.gates(tok)
            k = torch.full((Q,), kk, device=data.dev); m = relaxed_topk(g, k, cfg.tau_end, gen)
            soft.append(model.readout(pb, model.encode(b["q_emb"], b["q_pad"], b["v"], torch.log(m + 1e-6))[1]).cpu())
            hard.append(model.readout(pb, model.encode(b["q_emb"], b["q_pad"], b["v"], hard_bias(hard_topk(g, k)))[1]).cpu())
        s, h_ = torch.cat(soft), torch.cat(hard); res[str(kk)] = {"soft": float(s.mean()), "hard": float(h_.mean()), "gap": float((s - h_).mean())}
    model.train()
    return res


@torch.no_grad()
def dynamic_tau(model, data, cfg, target_fraction, chunk=64):
    """tau_dyn such that the mean kept fraction of 'smallest budget with r(top-k) <= tau' equals target_fraction on `data` (bisection)."""
    model.eval(); R = []
    grid = list(cfg.budgets) + [cfg.n_vis]
    for c in range(0, data.n, chunk):
        qi = np.arange(c, min(c + chunk, data.n)); b = data.batch(qi, np.zeros((len(qi), 1), dtype=int)); Q, N = len(qi), cfg.n_vis
        tok, pb = model.encode(b["q_emb"], b["q_pad"], b["v"], torch.zeros(Q, N, device=data.dev)); g = model.gates(tok); rows = []
        for kk in grid:
            k = torch.full((Q,), kk, device=data.dev); rows.append(model.readout(pb, model.encode(b["q_emb"], b["q_pad"], b["v"], hard_bias(hard_topk(g, k)))[1]).cpu())
        R.append(torch.stack(rows, 1))
    R = torch.cat(R).numpy(); fr = np.array(grid) / cfg.n_vis; model.train()
    def mean_frac(tau):
        ok = R <= tau; ok[:, -1] = True
        return float(fr[np.argmax(ok, 1)].mean())
    lo, hi = float(R.min()) - 1.0, float(R.max()) + 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mean_frac(mid) > target_fraction else (lo, mid)
    return {"tau": hi, "mean_kept_fraction": mean_frac(hi), "target": target_fraction}
