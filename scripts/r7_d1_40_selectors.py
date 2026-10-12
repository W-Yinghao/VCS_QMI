"""r7 R7-40b — the frozen D1-20 selector recipe on the second host (Qwen2.5-VL-7B), as an adapter around scripts/r7_d1_selectors.py (imported,
unchanged).  Only the host-interface differences of the plan (reports/R7/R7_40_PLAN_DRAFT.yaml) are changed:
  inputs     : merged Qwen visual tokens -> PCA 3584 -> 512 (fitted on FIT only, cache job), question = Qwen input-embedding lookup (3584)
  positions  : factorised learned row + column embeddings (max_rows × d, max_cols × d, N(0, 0.02²) init) indexed by the token's (row, col) on
               the image's own merged grid, instead of the learned 576 × d table of the 24 × 24 LLaVA grid
  variable N : each image keeps its own N tokens; a batch pads visual keys to the batch maximum with the additive key bias −1e9 (the frozen
               per-key bias mechanism), so padding affects nothing (unit-tested); gates and top-k act on valid tokens only
  budgets    : fractions f ∈ {0.125, 0.25, 0.5, 0.75} of each image's tokens, k = round(f·N) (r7_d1_40_masks.budget); the dynamic grid is
               f ∈ {0.125, 0.25, 0.5, 0.75, 1}
Everything else is the frozen D1-20 recipe: 2-layer pre-LN transformer d 512 / 8 heads / FFN 2048 / dropout 0.1, explicit fp32 attention,
attention pooling, critic / risk / gate heads, objectives S1 vcs / S2 logistic / S3 direct, phase A 20k steps of 64 questions × 4 of their 12
masks (AdamW, wd 0.01, 1k warm-up + cosine), CAL Spearman(r, h) selection every 1k steps with the native criterion logged, phase B 5k gate
steps (relaxed top-k with Gumbel noise, threshold midway between the k-th and (k+1)-th perturbed logits, tau cosine 1 -> 0.1, AdamW 3e-4),
soft / hard gap on CAL, dynamic tau by bisection on CAL, bit-exact resume.
Exact gradient accumulation (implementation, not recipe): with variable N the longest image of a step would pad every sequence; a step's
questions are therefore sorted by length and split into micro-batches whose explicit-attention size (sequences × padded length²) stays below
the frozen LLaVA step's size; each micro-batch adds its terms with the step's GLOBAL normalisers, so the loss and its gradient equal the
single-batch ones (unit-tested to fp32 precision with dropout off); only the dropout draws differ.  Evaluation passes are grouped the same way.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_d1_selectors as SEL  # noqa: E402  (frozen D1-20 module)

NEG = SEL.NEG
PAD_RANK = 32767                                                      # rank of padding tokens (never kept)
LLAVA_STEP_L = 576 + 30                                               # frozen LLaVA step: 576 visual + ~30 question tokens


@dataclass
class Cfg40(SEL.Cfg):
    emb_dim: int = 3584
    n_vis: int = 0                                                    # variable: each image's own N
    grid: int = 0
    budgets: tuple = ()                                               # token budgets are per image: see fractions
    fractions: tuple = (0.125, 0.25, 0.5, 0.75)
    max_rows: int = 64
    max_cols: int = 64
    cost_train: float = float(64 * 5 * LLAVA_STEP_L ** 2)             # frozen step: 64 full + 256 masked sequences at ~606 keys
    cost_eval: float = float(64 * 13 * LLAVA_STEP_L ** 2)             # frozen evaluation chunk: 64 questions × 13 sequences


class Selector40(SEL.Selector):
    """The frozen Selector with factorised (row, col) position tables and per-batch visual padding."""

    def __init__(self, cfg: Cfg40, objective: str):
        super().__init__(cfg, objective)
        del self.pos                                                                    # the 576 × d table of the frozen 24 × 24 grid
        self.row_pos = nn.Parameter(torch.randn(cfg.max_rows, cfg.d) * 0.02); self.col_pos = nn.Parameter(torch.randn(cfg.max_cols, cfg.d) * 0.02)

    def encode(self, q_emb, q_pad, v, vis_bias, rows, cols):
        """q_emb [B, Lq, E] (q_pad True = padding), v [B, N, Vd], vis_bias [B, N] additive (−1e9 on padding), rows / cols [B, N]."""
        xq = self.q_proj(q_emb) + self.typ[0]; xv = self.v_proj(v) + self.row_pos[rows] + self.col_pos[cols] + self.typ[1]
        qb = torch.where(q_pad, torch.full_like(q_pad, NEG, dtype=xv.dtype), torch.zeros_like(q_pad, dtype=xv.dtype))
        if self.cfg.no_question:                                                       # ablation A1 (not run on Qwen; kept for parity)
            qb = torch.full_like(qb, NEG)
        x = torch.cat([xq, xv], 1); kb = torch.cat([qb, vis_bias.to(xv.dtype)], 1)
        for blk in self.blocks:
            x = blk(x, kb[:, None, None, :])
        x = self.ln(x); w = torch.softmax((x @ self.pool_q) / math.sqrt(x.shape[-1]) + kb, 1)
        return x, (w[..., None] * x).sum(1)

    def gates40(self, tokens, n_vis: int):
        return self.gate(tokens[:, -n_vis:].detach()).squeeze(-1)


def full_bias(valid):
    return torch.where(valid, torch.zeros((), device=valid.device), torch.full((), NEG, device=valid.device))


GAMMA = 2 * 512 / (4 * 512 ** 2 + 2 * 512 * 2048)     # attention / linear multiply-adds per token and key (d 512, FFN 2048): cost ∝ L + γ L²
OVERHEAD = 1500.0                                      # fixed cost of one extra micro-batch (launches, sync), in padded-token units


def plan_groups(lengths, seqs_per_q: int, cost: float, gamma: float = GAMMA, overhead: float = OVERHEAD):
    """Positions of a step's questions, sorted by sequence length (stable), cut into consecutive micro-batches that minimise the padded
    compute Σ_g n_g·s·(L_g + γ·L_g²) + overhead·#groups (L_g = the group's longest length, s = sequences per question) subject to the memory
    cap n_g·s·L_g² <= cost (a single question always forms a group).  Exact dynamic programme over the sorted order; any partition gives the
    same loss and gradient (estimator_terms / gate_terms use the step's global normalisers), so this only sets speed and peak memory."""
    L = np.asarray(lengths, dtype=np.int64); order = np.argsort(L, kind="stable"); Ls = L[order].astype(np.float64); n = len(Ls)
    best = np.full(n + 1, np.inf); best[0] = 0.0; prev = np.zeros(n + 1, dtype=np.int64)
    for j in range(1, n + 1):                                                          # group = sorted positions i .. j-1, padded to Ls[j-1]
        Lj = Ls[j - 1]; m = j - np.arange(j)                                           # sizes for i = 0 .. j-1
        ok = (m == 1) | (m * seqs_per_q * Lj ** 2 <= cost)
        c = np.where(ok, best[:j] + m * seqs_per_q * (Lj + gamma * Lj ** 2) + overhead, np.inf); i = int(np.argmin(c))
        best[j], prev[j] = c[i], i
    groups, j = [], n
    while j > 0:
        i = int(prev[j]); groups.append(order[i:j]); j = i
    return groups[::-1]


def estimator_terms(model, b, wts, W_b, W_a):
    """This micro-batch's share of the frozen estimator loss, normalised by the step's global weight sums W_b (full) and W_a (masked)."""
    Q, M, Nb = b["keep"].shape; rep = lambda x: x.repeat_interleave(M, 0)  # noqa: E731
    _, pb = model.encode(b["q_emb"], b["q_pad"], b["v"], full_bias(b["valid"]), b["rows"], b["cols"])
    _, pa = model.encode(rep(b["q_emb"]), rep(b["q_pad"]), rep(b["v"]), SEL.hard_bias(b["keep"].reshape(Q * M, Nb)), rep(b["rows"]), rep(b["cols"]))
    C = b["C"].float(); w = torch.where(C > 0, wts[0], wts[1]); Cm, wm = rep(C), rep(w)
    if model.objective == "direct":
        y = (b["h"] if model.cfg.target == "h" else b["d"]).reshape(-1)
        return (wm * (model.readout(rep(pb), pa) - y) ** 2).sum() / W_a
    tb, fb = model.T(pb); ta, fa = model.T(pa)
    if model.objective == "vcs":
        jb = (w * torch.where(C > 0, SEL.a_plus(tb), SEL.a_minus(tb))).sum() / W_b
        ja = (wm * torch.where(Cm > 0, SEL.a_plus(ta), SEL.a_minus(ta))).sum() / W_a
        return -(jb + ja)
    yb, ya = (C > 0).float(), (Cm > 0).float()
    lb = (w * F.binary_cross_entropy_with_logits(2 * fb, yb, reduction="none")).sum() / W_b
    la = (wm * F.binary_cross_entropy_with_logits(2 * fa, ya, reduction="none")).sum() / W_a
    return lb + la


def relaxed_topk40(g, k, tau, valid, gen=None):
    """The frozen relaxed top-k over each row's VALID tokens: sigmoid((g + Gumbel − t_k) / tau), t_k midway between the k-th and (k+1)-th
    largest perturbed valid logits (detached), one unit below the smallest when k >= the row's valid count."""
    u = torch.rand(g.shape, generator=gen, device=g.device).clamp_(1e-9, 1 - 1e-9); z = g - torch.log(-torch.log(u))
    zs = torch.sort(torch.where(valid, z.detach(), torch.full_like(z, -float("inf"))), 1, descending=True).values; Nb = g.shape[1]
    nv = valid.sum(1); hi = zs.gather(1, (k - 1).clamp(0, Nb - 1)[:, None]); lo = zs.gather(1, k.clamp(max=Nb - 1)[:, None])
    t = torch.where((k >= nv)[:, None], hi - 1.0, 0.5 * (hi + lo))
    return torch.sigmoid((z - t) / tau)


def hard_topk40(g, k, valid):
    """Deployment mask: exactly k[b] highest gate logits among the row's valid tokens."""
    gm = torch.where(valid, g, torch.full_like(g, -float("inf"))); idx = torch.argsort(gm, 1, descending=True)
    ranks = torch.empty_like(idx); ranks.scatter_(1, idx, torch.arange(g.shape[1], device=g.device).expand_as(idx))
    return (ranks < k[:, None]) & valid


def gate_terms(model, b, k, tau, gen, Q_total):
    """This micro-batch's share of the frozen gate loss (mean predicted risk of the soft-masked observation over the step's questions)."""
    Nb = b["v"].shape[1]
    tok, pb = model.encode(b["q_emb"], b["q_pad"], b["v"], full_bias(b["valid"]), b["rows"], b["cols"])
    m = relaxed_topk40(model.gates40(tok, Nb), k, tau, b["valid"], gen)
    bias = torch.where(b["valid"], torch.log(m + 1e-6), torch.full_like(m, NEG))
    _, pa = model.encode(b["q_emb"], b["q_pad"], b["v"], bias, b["rows"], b["cols"])
    return model.readout(pb, pa).sum() / Q_total


class TupleData40:
    """R7-40a teacher records + cached features on one device.  feats [T, Vd] (all images' tokens, concatenated); per image offset, N and
    grid columns; per question: image, question ids, rank [n_q, 3, N_max] (rank of each token in the mechanism's ordering; PAD_RANK on
    padding), kk [n_q, 4] (k per fraction), C, h / d [n_q, 12] (mask order: mechanism-major × fractions, as the frozen D1-20 tuples)."""

    def __init__(self, feats, offsets, ntok, gcols, img_of, q_ids, q_len, emb_table, rank, kk, C, h, d, device, max_rows=64, max_cols=64):
        dev = device; self.dev = dev
        ntok_np, gcols_np = np.asarray(ntok), np.asarray(gcols); rows_needed = -(-ntok_np // np.maximum(gcols_np, 1))
        if (rows_needed > max_rows).any() or (gcols_np > max_cols).any():
            raise SystemExit(f"grid larger than the position tables ({int(rows_needed.max())} × {int(gcols_np.max())} > {max_rows} × {max_cols})")
        self.feats = torch.as_tensor(feats).to(dev); self.off = torch.as_tensor(np.asarray(offsets), device=dev)
        self.ntok = torch.as_tensor(ntok_np, device=dev); self.gcols = torch.as_tensor(gcols_np, device=dev)
        self.img_of = torch.as_tensor(np.asarray(img_of), device=dev)
        self.q_ids, self.q_len, self.emb = torch.as_tensor(q_ids, device=dev), torch.as_tensor(q_len, device=dev), emb_table.to(dev)
        self.rank, self.kk = torch.as_tensor(rank, device=dev), torch.as_tensor(kk, device=dev)
        self.C = torch.as_tensor(C, device=dev)
        self.h, self.d = torch.as_tensor(h, dtype=torch.float32, device=dev), torch.as_tensor(d, dtype=torch.float32, device=dev)
        self.n = len(C); self.lengths = np.asarray(q_len) + ntok_np[np.asarray(img_of)]      # selector sequence length per question

    def batch(self, qi, mi=None, all_masks=False):
        dev = self.dev; qi = torch.as_tensor(np.asarray(qi), device=dev); im = self.img_of[qi]; n = self.ntok[im]; Nb = int(n.max()); Q = len(qi)
        ar = torch.arange(Nb, device=dev); valid = ar[None, :] < n[:, None]
        idx = torch.where(valid, self.off[im][:, None] + ar[None, :], torch.zeros((), dtype=torch.long, device=dev))
        v = self.feats[idx].float() * valid[..., None]
        gc = self.gcols[im][:, None]; rows = torch.where(valid, ar[None, :] // gc, 0); cols = torch.where(valid, ar[None, :] % gc, 0)
        ids = self.q_ids[qi]; L = int(self.q_len[qi].max()); ids = ids[:, :L]; pad = torch.arange(L, device=dev)[None, :] >= self.q_len[qi][:, None]
        rank = self.rank[qi][:, :, :Nb].long(); kk = self.kk[qi].long(); n_f = kk.shape[1]
        if all_masks:
            keep = (rank[:, :, None, :] < kk[:, None, :, None]).reshape(Q, -1, Nb); h, d = self.h[qi], self.d[qi]
        else:
            mi = torch.as_tensor(np.asarray(mi), device=dev); rq = torch.arange(Q, device=dev)[:, None]
            keep = rank[rq, mi // n_f] < kk[rq, mi % n_f][..., None]; h, d = self.h[qi[:, None], mi], self.d[qi[:, None], mi]
        keep = keep & valid[:, None, :]
        return {"q_emb": self.emb[ids].float(), "q_pad": pad, "v": v, "valid": valid, "rows": rows, "cols": cols, "n": n, "kk": kk,
                "keep": keep, "C": self.C[qi], "h": h, "d": d}


@torch.no_grad()
def predict_r40(model, data, idx, with_critic=False, cost=None):
    """Predicted risk r(K) for every (question, mask) pair of the questions `idx` -> [len(idx), M] (+ critic logits from the same pass)."""
    model.eval(); idx = np.asarray(idx); M = data.h.shape[1]; cost = cost or model.cfg.cost_eval
    r = np.zeros((len(idx), M), np.float32); fb = np.zeros(len(idx), np.float32); fa = np.zeros((len(idx), M), np.float32)
    for g in plan_groups(data.lengths[idx], 1 + M, cost):
        b = data.batch(idx[g], all_masks=True); Q, M_, Nb = b["keep"].shape; rep = lambda x: x.repeat_interleave(M_, 0)  # noqa: E731
        _, pb = model.encode(b["q_emb"], b["q_pad"], b["v"], full_bias(b["valid"]), b["rows"], b["cols"])
        _, pa = model.encode(rep(b["q_emb"]), rep(b["q_pad"]), rep(b["v"]), SEL.hard_bias(b["keep"].reshape(Q * M_, Nb)), rep(b["rows"]), rep(b["cols"]))
        r[g] = model.readout(rep(pb), pa).reshape(Q, M_).float().cpu().numpy()
        if with_critic:
            fb[g] = model.T(pb)[1].float().cpu().numpy(); fa[g] = model.T(pa)[1].reshape(Q, M_).float().cpu().numpy()
    model.train()
    return (r, fb, fa) if with_critic else r


def k_of(data, qi, f):
    """k = round(f·N) for the questions qi (f = 1 keeps every token)."""
    n = data.ntok[data.img_of[torch.as_tensor(np.asarray(qi), device=data.dev)]]
    return n.clone() if f >= 1.0 else torch.clamp(torch.floor(f * n.double() + 0.5).long(), min=1)


@torch.no_grad()
def soft_hard_gap40(model, data, cfg, gen=None):
    """Per fraction: mean predicted risk of the relaxed mask at the final temperature vs the deployed hard top-k mask (CAL)."""
    model.eval(); res = {}; allq = np.arange(data.n)
    for f in cfg.fractions:
        soft, hard = np.zeros(data.n), np.zeros(data.n)
        for g in plan_groups(data.lengths, 3, cfg.cost_eval):
            b = data.batch(allq[g], np.zeros((len(g), 1), dtype=int)); Nb = b["v"].shape[1]; k = k_of(data, allq[g], f)
            tok, pb = model.encode(b["q_emb"], b["q_pad"], b["v"], full_bias(b["valid"]), b["rows"], b["cols"]); gg = model.gates40(tok, Nb)
            m = relaxed_topk40(gg, k, cfg.tau_end, b["valid"], gen); bias = torch.where(b["valid"], torch.log(m + 1e-6), torch.full_like(m, NEG))
            soft[g] = model.readout(pb, model.encode(b["q_emb"], b["q_pad"], b["v"], bias, b["rows"], b["cols"])[1]).cpu().numpy()
            hard[g] = model.readout(pb, model.encode(b["q_emb"], b["q_pad"], b["v"], SEL.hard_bias(hard_topk40(gg, k, b["valid"])), b["rows"], b["cols"])[1]).cpu().numpy()
        res[f"{f:g}"] = {"soft": float(soft.mean()), "hard": float(hard.mean()), "gap": float((soft - hard).mean())}
    model.train()
    return res


@torch.no_grad()
def risk_grid(model, data, cfg, qi=None):
    """Predicted risk r of the hard top-k mask at every fraction of the dynamic grid (cfg.fractions + 1) -> [n, 5]."""
    model.eval(); qi = np.arange(data.n) if qi is None else np.asarray(qi); grid = list(cfg.fractions) + [1.0]; R = np.zeros((len(qi), len(grid)))
    for g in plan_groups(data.lengths[qi], 1 + len(grid), cfg.cost_eval):
        b = data.batch(qi[g], np.zeros((len(g), 1), dtype=int)); Nb = b["v"].shape[1]
        tok, pb = model.encode(b["q_emb"], b["q_pad"], b["v"], full_bias(b["valid"]), b["rows"], b["cols"]); gg = model.gates40(tok, Nb)
        for j, f in enumerate(grid):
            keep = hard_topk40(gg, k_of(data, qi[g], f), b["valid"])
            R[g, j] = model.readout(pb, model.encode(b["q_emb"], b["q_pad"], b["v"], SEL.hard_bias(keep), b["rows"], b["cols"])[1]).float().cpu().numpy()
    model.train()
    return R


def dynamic_tau40(model, data, cfg, target_fraction):
    """tau_dyn such that the mean kept fraction of 'smallest fraction in {0.125, 0.25, 0.5, 0.75, 1} with r(top-k) <= tau' equals the target
    on `data` (bisection, as the frozen dynamic_tau; fractions are nominal, k = round(f·N))."""
    R = risk_grid(model, data, cfg); fr = np.array(list(cfg.fractions) + [1.0])

    def mean_frac(tau):
        ok = R <= tau; ok[:, -1] = True
        return float(fr[np.argmax(ok, 1)].mean())
    lo, hi = float(R.min()) - 1.0, float(R.max()) + 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mean_frac(mid) > target_fraction else (lo, mid)
    return {"tau": hi, "mean_kept_fraction": mean_frac(hi), "target": target_fraction}


def train40(cfg: Cfg40, objective: str, fit: TupleData40, cal: TupleData40, wts, log_path: Path | None = None, ckpt_path: Path | None = None,
            resume_path: Path | None = None, stop_after: int | None = None):
    """Phase A (estimator, CAL-selected) then phase B (gates, fixed steps) — the frozen SEL.train loop with the variable-N batches and
    exact micro-batch accumulation.  Resume as frozen (state at every evaluation and at the end of phase A; bit-exact continuation)."""
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); dev = fit.dev; gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    model = Selector40(cfg, objective).to(dev); wts = torch.as_tensor(wts, dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(model.estimator_parameters(), lr=cfg.lr, weight_decay=cfg.wd); rng = np.random.default_rng(cfg.seed)
    M = fit.h.shape[1]; cal_idx = np.arange(cal.n); cal_h = cal.h.cpu().numpy(); log = {"cfg": asdict(cfg), "objective": objective, "evals": [], "host": "Qwen2.5-VL-7B"}
    best = (-np.inf, None, 0); t0 = time.time(); start, phase_a_done = 1, False
    if resume_path is not None and Path(resume_path).exists():
        ck = torch.load(resume_path, map_location=dev, weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        best, log, start, phase_a_done = ck["best"], ck["log"], ck["step"] + 1, ck.get("phase_a_done", False); SEL._set_rng_state(ck["rng"], rng, gen)
        log.setdefault("resumed_at", []).append(ck["step"])

    def save(step, done=False):
        if resume_path is not None:
            tmp = Path(str(resume_path) + ".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "best": best, "log": log, "step": step, "phase_a_done": done,
                        "rng": SEL._rng_state(rng, gen)}, tmp)
            tmp.replace(resume_path)
    tm = log.setdefault("timing", {"phase_a_train_s": 0.0, "phase_a_steps": 0, "eval_s": 0.0, "eval_pairs": 0, "phase_b_s": 0.0, "phase_b_steps": 0,
                                   "micro_batches_a": 0, "micro_batches_b": 0})
    cuda = dev != "cpu" and torch.cuda.is_available(); sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)  # noqa: E731
    if not phase_a_done:
        for step in range(start, cfg.steps_a + 1):
            sync(); ts = time.time()
            for gpar in opt.param_groups:
                gpar["lr"] = SEL.cosine_lr(step, cfg.steps_a, cfg.lr, min(cfg.warmup, cfg.steps_a))
            qi = rng.choice(fit.n, size=min(cfg.q_per_step, fit.n), replace=False); mi = np.stack([rng.choice(M, size=cfg.masks_per_q, replace=False) for _ in qi])
            Cq = fit.C[torch.as_tensor(qi, device=dev)].float(); W_b = torch.where(Cq > 0, wts[0], wts[1]).sum(); W_a = cfg.masks_per_q * W_b
            opt.zero_grad(set_to_none=True); loss = 0.0; groups = plan_groups(fit.lengths[qi], 1 + cfg.masks_per_q, cfg.cost_train)
            for g in groups:
                lg = estimator_terms(model, fit.batch(qi[g], mi[g]), wts, W_b, W_a); lg.backward(); loss += float(lg.detach())
            opt.step(); sync(); tm["phase_a_train_s"] += time.time() - ts; tm["phase_a_steps"] += 1; tm["micro_batches_a"] += len(groups)
            if step % cfg.eval_every == 0 or step == cfg.steps_a:
                ts = time.time(); r, fb, fa = predict_r40(model, cal, cal_idx, with_critic=True); sync(); tm["eval_s"] += time.time() - ts; tm["eval_pairs"] += r.size
                rho = SEL.spearman(r.reshape(-1), cal_h.reshape(-1))
                nat = SEL.native_criterion(objective, cfg.target, r, fb, fa, cal.C.cpu().numpy(), cal_h, cal.d.cpu().numpy(), wts.cpu().numpy())
                log["evals"].append({"step": step, "loss": float(loss), "cal_spearman_r_h": rho, "native": nat, "seconds": time.time() - t0})
                if np.isfinite(rho) and rho > best[0]:
                    best = (rho, {k: v.detach().clone() for k, v in model.state_dict().items()}, step)
                save(step)
                if stop_after is not None and step >= stop_after:
                    return model, log
        if best[1] is not None:
            model.load_state_dict(best[1])
        log["selected_step"], log["selected_cal_spearman"] = best[2], best[0]
        ev = [e for e in log["evals"] if "native" in e]                                # secondary: the native-criterion choice, reported only
        if ev:
            pick = (max if ev[0]["native"]["better"] == "higher" else min)(ev, key=lambda e: e["native"]["value"])
            log["native_selected"] = {"criterion": pick["native"]["name"], "step": pick["step"], "native_value": pick["native"]["value"],
                                      "cal_spearman_r_h": pick["cal_spearman_r_h"], "role": "secondary (reported; never used for selection)"}
        save(cfg.steps_a, done=True)
    for p in model.estimator_parameters():                                             # phase B: estimator frozen
        p.requires_grad_(False)
    gopt = torch.optim.AdamW(model.gate.parameters(), lr=cfg.lr_gate, weight_decay=cfg.wd); nfr = len(cfg.fractions)
    gl = None; sync(); tb0 = time.time()
    for step in range(1, cfg.steps_b + 1):
        tau = cfg.tau_end + 0.5 * (cfg.tau_start - cfg.tau_end) * (1 + math.cos(math.pi * step / max(cfg.steps_b, 1)))
        qi = rng.choice(fit.n, size=min(cfg.q_per_step, fit.n), replace=False)
        kidx = torch.randint(nfr, (len(qi),), generator=gen, device=dev); k = fit.kk[torch.as_tensor(qi, device=dev), kidx].long()
        gopt.zero_grad(set_to_none=True); gl = 0.0; groups = plan_groups(fit.lengths[qi], 2, cfg.cost_train)
        for g in groups:
            lg = gate_terms(model, fit.batch(qi[g], np.zeros((len(g), 1), dtype=int)), k[torch.as_tensor(g, device=dev)], tau, gen, len(qi))
            lg.backward(); gl += float(lg.detach())
        gopt.step(); tm["micro_batches_b"] += len(groups)
    sync(); tm["phase_b_s"] += time.time() - tb0; tm["phase_b_steps"] += cfg.steps_b
    log["gate_final_loss"] = float(gl) if gl is not None else None
    log["soft_hard_gap"] = soft_hard_gap40(model, cal, cfg, gen) if cfg.steps_b else None
    log["seconds"] = time.time() - t0
    if cuda:
        log["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    if log_path:
        Path(log_path).write_text(json.dumps(log, indent=1))
    if ckpt_path:
        torch.save({"state_dict": model.state_dict(), "cfg": asdict(cfg), "objective": objective, "host": "Qwen2.5-VL-7B"}, ckpt_path)
    return model, log
