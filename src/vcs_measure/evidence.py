"""P122 — v6 V6-EVIDENCE: hierarchical measurement critics s -> Z -> H on frozen encoders under a COMMON measurement augmentation.

Pairs (per split, base images disjoint across FIT / TUNE / EVAL; half of each split are anchors, half independent partners):
    P_i = (view A1 of anchor i, view A2 of anchor i)          Q_i = (view A1 of anchor i, view B of partner i)
P_i and Q_i share the anchor view A1; the evaluation unit is the pair index i (one P and one Q per unit).  The augmented images are generated once
per (dataset, measurement law, pool) from fixed per-image seeds and shared by every encoder, so encoders differ only in the representation.

Critics return the pre-tanh logit f; T = tanh f.  Losses on the SAME logits:
    vcs : -J(T) = -[mean_P (T - T^2/2) + mean_Q (-T - T^2/2)]
    js  : mean_P softplus(-2f) + mean_Q softplus(2f)      (logistic logit 2f; common posterior T = tanh f)
Levels (each level's selection only by TUNE own-objective risk; the parent is always a candidate):
    train : the run's own training scorer, unchanged (VCS runs; SimCLR has none)
    s     : affine tanh, 16 / 32 equal-frequency-bin squared regression, 1-D MLP 32-32 on s = <z1, z2>   (+ train as a candidate)
    Z     : f_Z = f_s(s) + q_psi(z1, z2)   (selected s branch frozen; q zero-initialised; "no residual" is a candidate)
    H     : f_H = f_Z(z1, z2) + q_omega(h1, h2)  (selected Z model frozen; h standardised with FIT statistics, a fixed deterministic map)
Residual input [u, v, u*v, |u-v|], symmetrised q = 1/2 (q(u, v) + q(v, u)) (same-modality pairs).  EVAL is only read for reporting.
Reported increments (J_s - J_train, J_Z - J_s, J_H - J_Z) are "additional score recovered in these function classes and budgets", not true terms.
"""
from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T

CODE_VERSION = "p122-evidence-1"
LRS = (1e-4, 5e-4, 2e-3)
INITS = (0, 1, 2)
STEPS = 1000
EVAL_EVERY = 10
WD = 1e-2
CLIP = 1.0 - 1e-6


# ----------------------------------------------------------------------------------------------------------------------- pools
def make_pools(fit_uids: np.ndarray, sizes: dict[str, int], seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Disjoint base-image pools from the manifest FIT uids; each pool split into anchors (first half) and partners (second half) of a fixed
    permutation.  Returns {split: {"anchors": uids, "partners": uids}} with len(anchors) == len(partners) == size / 2."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(fit_uids, dtype=np.int64))
    out, i = {}, 0
    for k, n in sizes.items():
        assert n % 2 == 0, "pool sizes must be even (anchor / partner halves)"
        block = perm[i:i + n]; i += n
        out[k] = {"anchors": block[: n // 2].copy(), "partners": block[n // 2:].copy()}
    assert i <= len(perm), f"pool sizes {sizes} exceed the {len(perm)} available FIT images"
    allu = np.concatenate([np.concatenate([v["anchors"], v["partners"]]) for v in out.values()])
    assert len(np.unique(allu)) == len(allu), "pools / roles overlap"
    return out


def pools_hash(pools: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(pools):
        for r in ("anchors", "partners"):
            h.update(k.encode()); h.update(r.encode()); h.update(np.ascontiguousarray(pools[k][r]).tobytes())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------------------------------------------------- augmentation
def stochastic_transform(views: dict) -> T.Compose:
    """The random part of the training two-view transform (no ToTensor / Normalize): RRC, flip, colour jitter (apply-p), grayscale.
    Raises on blur / solarize so the measurement law is exactly the documented block."""
    assert float(views.get("gaussian_blur_p", 0.0)) == 0.0 and float(views.get("solarize_p", 0.0)) == 0.0
    rrc, cj = views["random_resized_crop"], views["color_jitter"]
    interp = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC}[rrc["interpolation"]]
    return T.Compose([
        T.RandomResizedCrop(rrc["size"], scale=tuple(rrc["scale"]), ratio=tuple(rrc["ratio"]), interpolation=interp, antialias=rrc["antialias"]),
        T.RandomHorizontalFlip(p=views["horizontal_flip_p"]),
        T.RandomApply([T.ColorJitter(cj["brightness"], cj["contrast"], cj["saturation"], cj["hue"])], p=cj["p"]),
        T.RandomGrayscale(p=views["grayscale_p"]),
    ])


def law_signature(views: dict) -> str:
    keys = ("random_resized_crop", "horizontal_flip_p", "color_jitter", "grayscale_p", "gaussian_blur_p", "solarize_p")
    return hashlib.sha256(repr({k: views.get(k) for k in keys}).encode()).hexdigest()[:16]


def augment(images_uint8: np.ndarray, uids: np.ndarray, views: dict, seed: int, tag: int) -> torch.Tensor:
    """uint8 [n, 3, 32, 32]; image k uses torch seed (seed, tag, uid) under fork_rng, so it is reproducible and independent of n and order."""
    tf = stochastic_transform(views); out = torch.empty((len(uids), 3, 32, 32), dtype=torch.uint8)
    with torch.random.fork_rng(devices=[]):
        for k, u in enumerate(np.asarray(uids, dtype=np.int64)):
            torch.manual_seed((int(seed) * 1_000_003 + int(tag) * 100_003 + int(u)) % (2 ** 63 - 1))
            out[k] = T.functional.pil_to_tensor(tf(Image.fromarray(images_uint8[int(u)])))
    return out


def augment_pools(images_uint8: np.ndarray, pools: dict, views: dict, seed: int) -> dict[str, dict[str, torch.Tensor]]:
    """Per split: A1 / A2 = two views of the anchors (tags 1, 2), B = one view of the partners (tag 3)."""
    return {k: {"A1": augment(images_uint8, v["anchors"], views, seed, 1), "A2": augment(images_uint8, v["anchors"], views, seed, 2),
                "B": augment(images_uint8, v["partners"], views, seed, 3)} for k, v in pools.items()}


# ----------------------------------------------------------------------------------------------------------------------- features
@torch.no_grad()
def encode(R: dict, imgs: torch.Tensor, mean, std, device, batch: int = 1024) -> dict[str, torch.Tensor]:
    """Frozen eval-mode encoder / projector; float32 raw h and z = L2(projector(h)) (the critic input of the VCS runs)."""
    from vcs_measure.common import layer_features
    m = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1); s = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    hs, zs = [], []
    for i in range(0, len(imgs), batch):
        x = (imgs[i:i + batch].to(device).float() / 255.0 - m) / s
        f = layer_features(R["encoder"], R["projector"], x, R["eps"])
        hs.append(f["h"].float().cpu()); zs.append(f["z"].float().cpu())
    return {"h": torch.cat(hs), "z": torch.cat(zs)}


# ----------------------------------------------------------------------------------------------------------------------- objectives
def j_terms(fp: torch.Tensor, fq: torch.Tensor) -> torch.Tensor:
    """Per-unit J contribution j_i = (T_P - T_P^2/2) + (-T_Q - T_Q^2/2); J = mean(j_i) (balanced units)."""
    tp, tq = torch.tanh(fp), torch.tanh(fq)
    return (tp - 0.5 * tp ** 2) + (-tq - 0.5 * tq ** 2)


def own_risk(fp: torch.Tensor, fq: torch.Tensor, loss: str) -> torch.Tensor:
    if loss == "vcs":
        return -j_terms(fp, fq).mean()
    if loss == "js":
        return F.softplus(-2.0 * fp).mean() + F.softplus(2.0 * fq).mean()
    raise ValueError(loss)


def readouts(fp: torch.Tensor, fq: torch.Tensor) -> dict[str, float]:
    fp, fq = fp.double(), fq.double(); J = float(j_terms(fp, fq).mean())
    js_native = float((-F.softplus(-2.0 * fp)).mean() - F.softplus(2.0 * fq).mean() + math.log(4.0))
    return {"J": J, "sq_risk": 1.0 - J, "js_native": js_native}


# ----------------------------------------------------------------------------------------------------------------------- critics
class Pairs:
    """Features of one split: anchor view A1, anchor view A2, partner view B (rows aligned by unit i)."""

    def __init__(self, z1, z2, zb, h1=None, h2=None, hb=None):
        self.z1, self.z2, self.zb, self.h1, self.h2, self.hb = z1, z2, zb, h1, h2, hb

    def to(self, device):
        mv = lambda t: None if t is None else t.to(device)
        return Pairs(mv(self.z1), mv(self.z2), mv(self.zb), mv(self.h1), mv(self.h2), mv(self.hb))

    def s(self):
        return (self.z1 * self.z2).sum(1), (self.z1 * self.zb).sum(1)


class TrainScorer:
    """The run's own training critic, unchanged: f = critic.logits(z1, z2)."""
    kind = "train"

    def __init__(self, critic):
        self.c = critic

    @torch.no_grad()
    def __call__(self, P: Pairs):
        return self.c.logits(P.z1, P.z2).float(), self.c.logits(P.z1, P.zb).float()


class ScalarFn:
    """f = g(s) for a fitted scalar map g (affine / bins / MLP)."""
    kind = "s"

    def __init__(self, g: Callable, name: str, params: dict):
        self.g, self.name, self.params = g, name, params

    def __call__(self, P: Pairs):
        sp, sq = P.s(); return self.g(sp), self.g(sq)

    def on_s(self, s):
        return self.g(s)


def fit_affine(Pf: Pairs, Pt: Pairs, loss: str) -> list[dict]:
    sp, sq = (t.double() for t in Pf.s()); out = []
    for a0 in (1.0, 5.0, 20.0):
        t0 = time.time(); ab = torch.tensor([a0, 0.0], dtype=torch.float64, device=sp.device, requires_grad=True)
        opt = torch.optim.LBFGS([ab], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); r = own_risk(ab[0] * sp + ab[1], ab[0] * sq + ab[1], loss); r.backward(); return r
        opt.step(closure)
        a, b = float(ab[0].detach()), float(ab[1].detach())
        g = lambda s, a=a, b=b: (a * s + b).float()
        fn = ScalarFn(g, "affine", {"a": a, "b": b, "a_init": a0})
        out.append({"fn": fn, "seconds": time.time() - t0})
    return out


def fit_bins(Pf: Pairs, K: int) -> dict:
    """Equal-frequency bins of s on FIT (P and Q pooled); per bin T = (n_P - n_Q) / (n_P + n_Q) (balanced: equal numbers of P and Q).
    This is the in-bin optimum of BOTH losses (squared risk and logistic), so the candidate is loss-independent."""
    t0 = time.time(); sp, sq = Pf.s(); s_all = torch.cat([sp, sq])
    edges = torch.quantile(s_all.double(), torch.linspace(0, 1, K + 1, dtype=torch.float64, device=s_all.device))[1:-1].float()
    bp = torch.bucketize(sp, edges); bq = torch.bucketize(sq, edges)
    npos = torch.bincount(bp, minlength=K).double(); nneg = torch.bincount(bq, minlength=K).double()
    tb = ((npos - nneg) / (npos + nneg).clamp_min(1.0)).clamp(-CLIP, CLIP)
    fb = torch.atanh(tb).float()
    g = lambda s, edges=edges, fb=fb: fb[torch.bucketize(s, edges)]
    return {"fn": ScalarFn(g, f"bins{K}", {"K": K, "empty_bins": int(((npos + nneg) == 0).sum())}), "seconds": time.time() - t0}


class MLP1D(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, s):
        return self.net(s.unsqueeze(1)).squeeze(1)


class SymResidual(nn.Module):
    """q(u, v) on [u, v, u*v, |u-v|], hidden 256 x 2, last layer zero-initialised (step 0 = parent); symmetrised 1/2 (q(u,v) + q(v,u))."""

    def __init__(self, d: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def _q(self, u, v):
        return self.net(torch.cat([u, v, u * v, (u - v).abs()], 1)).squeeze(1)

    def forward(self, u, v):
        return 0.5 * (self._q(u, v) + self._q(v, u))


class NestedFn:
    """f = parent(P) + q(rep1, rep2) on representation 'z' or 'h' (h standardised with FIT mean / sd); parent frozen."""

    def __init__(self, parent, q: SymResidual | None, rep: str, mu=None, sd=None, name: str = ""):
        self.parent, self.q, self.rep, self.mu, self.sd, self.name = parent, q, rep, mu, sd, name
        self.kind = rep.upper()

    def _x(self, t):
        return t if self.rep == "z" else (t - self.mu) / self.sd

    def __call__(self, P: Pairs, parent_cache=None):
        fp0, fq0 = parent_cache if parent_cache is not None else self.parent(P)
        if self.q is None:
            return fp0, fq0
        a1, a2, ab = (P.z1, P.z2, P.zb) if self.rep == "z" else (P.h1, P.h2, P.hb)
        return fp0 + self.q(self._x(a1), self._x(a2)), fq0 + self.q(self._x(a1), self._x(ab))


def _train(model_fn: Callable, params, Pf, Pt, loss, lr, seed, steps=None, every=None, fit_cache=None, tune_cache=None):
    """AdamW full-batch on FIT, TUNE own risk every `every` steps; returns best state (step 0 included) and its TUNE risk."""
    steps = STEPS if steps is None else steps; every = EVAL_EVERY if every is None else every  # module globals read at call time (smoke overrides)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WD)
    with torch.no_grad():
        best = float(own_risk(*model_fn(Pt, tune_cache), loss)); best_step = 0
    best_state = [p.detach().clone() for p in params]
    for step in range(1, steps + 1):
        opt.zero_grad(); r = own_risk(*model_fn(Pf, fit_cache), loss); r.backward(); opt.step()
        if step % every == 0 or step == steps:
            with torch.no_grad():
                v = float(own_risk(*model_fn(Pt, tune_cache), loss))
            if v < best:
                best, best_step, best_state = v, step, [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, b in zip(params, best_state):
            p.copy_(b)
    return best, best_step


def fit_mlp1d(Pf: Pairs, Pt: Pairs, loss: str, device) -> list[dict]:
    out = []
    for lr in LRS:
        for init in INITS:
            t0 = time.time(); torch.manual_seed(10_000 + init); m = MLP1D().to(device)
            fn = lambda P, cache=None, m=m: (m(P.s()[0]), m(P.s()[1]))
            tune_risk, step = _train(fn, list(m.parameters()), Pf, Pt, loss, lr, init)
            m.eval()
            g = lambda s, m=m: m(s).detach()
            out.append({"fn": ScalarFn(g, "mlp1d", {"lr": lr, "init": init, "best_step": step}), "seconds": time.time() - t0})
    return out


def fit_nested(parent, Pf: Pairs, Pt: Pairs, loss: str, rep: str, device, mu=None, sd=None) -> list[dict]:
    """Candidates: the parent itself (no residual) + LRS x INITS residual fits, each early-stopped on TUNE (step 0 = parent included)."""
    with torch.no_grad():
        fc, tc = parent(Pf), parent(Pt)
        fc = tuple(t.detach() for t in fc); tc = tuple(t.detach() for t in tc)
    out = [{"fn": NestedFn(parent, None, rep, mu, sd, name=f"{rep}:none"), "seconds": 0.0, "params": {"residual": False}}]
    d = (Pf.z1 if rep == "z" else Pf.h1).shape[1]
    for lr in LRS:
        for init in INITS:
            t0 = time.time(); torch.manual_seed(20_000 + init); q = SymResidual(d).to(device)
            nf = NestedFn(parent, q, rep, mu, sd, name=f"{rep}:residual")
            fn = lambda P, cache, nf=nf: nf(P, cache)
            tune_risk, step = _train(fn, list(q.parameters()), Pf, Pt, loss, lr, init, fit_cache=fc, tune_cache=tc)
            q.eval()
            out.append({"fn": nf, "seconds": time.time() - t0, "params": {"residual": True, "lr": lr, "init": init, "best_step": step}})
    return out


# ----------------------------------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def tune_risk_of(fn, Pt, loss):
    return float(own_risk(*fn(Pt), loss))


@torch.no_grad()
def unit_j(fn, P):
    fp, fq = fn(P); return j_terms(fp.double(), fq.double()).cpu().numpy(), fp, fq


def boot_increment(jb: np.ndarray, ja: np.ndarray, reps: int = 1000, seed: int = 7) -> dict:
    d = jb - ja; rng = np.random.default_rng(seed); n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(reps)])
    return {"mean": float(d.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "units": int(n)}


@torch.no_grad()
def density_diag(fn, P: Pairs, n_anchor: int = 512, n_partner: int = 512) -> dict:
    """Necessary-condition diagnostics (f = 0 passes both): global log E_Q e^{2f} on the EVAL Q pairs, and per-anchor log E_{Z2 ~ P_Z2} e^{2f(z1, Z2)}
    with the independent partner views B as the marginal pool (log-sum-exp), with the effective sample size of the weights."""
    fp, fq = fn(P); nq = len(fq)
    glob = float(torch.logsumexp(2.0 * fq.double(), 0) - math.log(nq))
    na, nb = min(n_anchor, len(P.z1)), min(n_partner, len(P.zb))
    vals, ess = [], []
    for i in range(na):
        rep = lambda t: None if t is None else t[i:i + 1].expand(nb, -1)
        Pi = Pairs(rep(P.z1), P.zb[:nb], P.zb[:nb], rep(P.h1), None if P.hb is None else P.hb[:nb], None if P.hb is None else P.hb[:nb])
        f = fn(Pi)[1].double()
        lse = torch.logsumexp(2.0 * f, 0); vals.append(float(lse - math.log(nb)))
        w = torch.exp(2.0 * f - lse); ess.append(float(1.0 / (w ** 2).sum()))
    v = np.array(vals)
    return {"global_log_mean_exp_2f_Q": glob, "per_anchor_mean": float(v.mean()), "per_anchor_sd": float(v.std()),
            "per_anchor_ess_mean": float(np.mean(ess)), "n_anchor": na, "n_partner": nb,
            "note": "necessary-condition diagnostic only (0 for a correct full density ratio; f = 0 also gives 0)"}


def select(cands: list[dict], Pt: Pairs, loss: str) -> tuple[dict, list[dict]]:
    rows = []
    for c in cands:
        c["tune_risk"] = tune_risk_of(c["fn"], Pt, loss); rows.append(c)
    best = min(rows, key=lambda c: c["tune_risk"])
    return best, rows
