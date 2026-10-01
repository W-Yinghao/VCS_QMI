"""X1 — an independent measurement-critic family on frozen two-view pairs (package v4 §X1).

P = (view a of image i, view b of image i); Q = (view a of image i, view b of image j), j = a cyclic shift (K partners, nonzero) inside the same split.
Every critic is fitted on FIT pairs only, its hyper-parameters / step / class are chosen on TUNE, and EVAL J is recorded unclipped (it can be < 0).
J(T) = mean_P(T − T²/2) + mean_Q(−T − T²/2), T = tanh(f) bounded in (−1, 1).  Measurement J is a finite-class lower readout, never called S.

Candidates
    zero        T ≡ 0 (J = 0): lets the selection say "no usable estimate"
    cosine      calibrated angular critic f = a<u1, u2> + b on L2-normalised features, (a, b) fitted on FIT by L-BFGS on −J
    mlp         symmetric pair MLP on [u1 ⊙ u2, |u1 − u2|] (hidden 256, 2 ReLU layers), AdamW, early stop on TUNE J
    prod_ridge  closed-form ridge of the quadratic score in the class phi = [u1 ⊙ u2, 1] (w = ½ (A_M + lam I)^-1 d), tanh(c phi w), lam and c on TUNE
    rff_ridge   the same closed form on random Fourier features of u1 ⊙ u2 (D = 1024, Gaussian kernel, bandwidth = median heuristic × {0.5, 1, 2})
"""
from __future__ import annotations

import math
import time
from typing import Any

import torch
from torch import nn


def j_value(tp: torch.Tensor, tq: torch.Tensor) -> float:
    return float((tp - 0.5 * tp ** 2).mean() + (-tq - 0.5 * tq ** 2).mean())


def make_pairs(n: int, k: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Q partner indices: for each of K draws a random nonzero cyclic shift (the recipe's product sampler); returns (anchor idx, partner idx)."""
    shifts = torch.randperm(n - 1, generator=gen)[:k] + 1
    a = torch.arange(n).repeat(k); b = torch.cat([(torch.arange(n) + int(s)) % n for s in shifts])
    return a, b


class PairSet:
    """Two-view features of one split: u1, u2 [n, d] (same base image per row) and the Q partner index lists."""

    def __init__(self, u1: torch.Tensor, u2: torch.Tensor, k: int, seed: int, block: int | None = None):
        """``block`` = number of base images per augmentation draw when several draws are stacked: Q partners are formed inside each draw
        block only, so a Q pair never joins two augmentations of the same base image."""
        g = torch.Generator().manual_seed(seed)
        self.u1, self.u2 = u1, u2
        block = block or len(u1)
        assert len(u1) % block == 0
        qa, qb = [], []
        for s in range(0, len(u1), block):
            a, b = make_pairs(block, k, g); qa.append(a + s); qb.append(b + s)
        self.qa, self.qb = torch.cat(qa), torch.cat(qb)

    def p(self):
        return self.u1, self.u2

    def q(self):
        return self.u1[self.qa], self.u2[self.qb]


# ---------------------------------------------------------------------------------------------------------------------- candidates
def fit_cosine(fit: PairSet, tune: PairSet) -> dict[str, Any]:
    n1 = lambda x: nn.functional.normalize(x, dim=1)
    sp = (n1(fit.u1) * n1(fit.u2)).sum(1).double(); qa, qb = fit.q(); sq = (n1(qa) * n1(qb)).sum(1).double()
    best = None
    for a0 in (1.0, 5.0, 20.0):
        ab = torch.tensor([a0, 0.0], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([ab], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            tp, tq = torch.tanh(ab[0] * sp + ab[1]), torch.tanh(ab[0] * sq + ab[1])
            loss = -((tp - 0.5 * tp ** 2).mean() + (-tq - 0.5 * tq ** 2).mean()); loss.backward(); return loss
        opt.step(closure)
        a, b = float(ab[0].detach()), float(ab[1].detach())
        f = lambda x1, x2, a=a, b=b: torch.tanh(a * (n1(x1) * n1(x2)).sum(1) + b)
        tj = j_value(f(*tune.p()), f(*tune.q()))
        if best is None or tj > best["tune_J"]:
            best = {"fn": f, "tune_J": tj, "params": {"a": a, "b": b, "a_init": a0}}
    return best


class _PairMLP(nn.Module):
    def __init__(self, d: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x1, x2):
        return self.net(torch.cat((x1 * x2, (x1 - x2).abs()), 1)).squeeze(1)


def fit_mlp(fit: PairSet, tune: PairSet, *, steps: int = 1500, lr: float = 1e-3, wd: float = 1e-2, batch: int = 4096, seed: int = 0,
            device: torch.device | str = "cpu") -> dict[str, Any]:
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    m = _PairMLP(fit.u1.shape[1]).to(device); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    P1, P2 = (x.to(device) for x in fit.p()); Q1, Q2 = (x.to(device) for x in fit.q())
    T1, T2 = (x.to(device) for x in tune.p()); U1, U2 = (x.to(device) for x in tune.q())
    best, best_state, best_step = -float("inf"), None, 0
    for step in range(1, steps + 1):
        ip = torch.randint(len(P1), (min(batch, len(P1)),), generator=g).to(device); iq = torch.randint(len(Q1), (min(batch, len(Q1)),), generator=g).to(device)
        tp, tq = torch.tanh(m(P1[ip], P2[ip])), torch.tanh(m(Q1[iq], Q2[iq]))
        loss = -((tp - 0.5 * tp ** 2).mean() + (-tq - 0.5 * tq ** 2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == steps:
            with torch.no_grad():
                tj = j_value(torch.tanh(m(T1, T2)), torch.tanh(m(U1, U2)))
            if tj > best:
                best, best_step, best_state = tj, step, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    f = lambda x1, x2: torch.tanh(m(x1.to(device), x2.to(device))).cpu()
    return {"fn": f, "tune_J": best, "params": {"best_step": best_step, "steps": steps}}


def _ridge_closed(phi_p, phi_q, tphi_p, tphi_q, lams=(1e-3, 1e-2, 1e-1, 1.0), cs=None) -> dict[str, Any]:
    """w* = ½ (A_M + lam·sc·I)^-1 (mean phi_P − mean phi_Q), A_M = ½(E_P phi phiᵀ + E_Q phi phiᵀ); T = tanh(c phi w); lam, c chosen on TUNE J."""
    cs = cs if cs is not None else torch.logspace(-1, 1.5, 25).tolist()
    d = phi_p.mean(0) - phi_q.mean(0); A = 0.5 * (phi_p.T @ phi_p / len(phi_p) + phi_q.T @ phi_q / len(phi_q)); sc = float(torch.diag(A).mean())
    best = None
    for lam in lams:
        w = 0.5 * torch.linalg.solve(A + lam * sc * torch.eye(len(A), dtype=A.dtype), d)
        sp, sq = tphi_p @ w, tphi_q @ w
        for c in cs:
            tj = j_value(torch.tanh(c * sp), torch.tanh(c * sq))
            if best is None or tj > best["tune_J"]:
                best = {"w": w, "c": float(c), "lam": lam, "tune_J": tj}
    return best


def fit_prod_ridge(fit: PairSet, tune: PairSet) -> dict[str, Any]:
    phi = lambda x1, x2: torch.cat(((x1 * x2).double(), torch.ones(len(x1), 1, dtype=torch.float64)), 1)
    b = _ridge_closed(phi(*fit.p()), phi(*fit.q()), phi(*tune.p()), phi(*tune.q()))
    f = lambda x1, x2, w=b["w"], c=b["c"]: torch.tanh(c * (phi(x1, x2) @ w)).float()
    return {"fn": f, "tune_J": b["tune_J"], "params": {"lam": b["lam"], "c": b["c"]}}


def fit_rff_ridge(fit: PairSet, tune: PairSet, *, D: int = 1024, seed: int = 0) -> dict[str, Any]:
    x = (fit.u1 * fit.u2).double(); g = torch.Generator().manual_seed(seed)
    sub = x[torch.randperm(len(x), generator=g)[:2000]]; med = float(torch.cdist(sub, sub).median())
    best = None
    for mult in (0.5, 1.0, 2.0):
        bw = max(med * mult, 1e-6)
        W = torch.randn(x.shape[1], D, generator=g, dtype=torch.float64) / bw; bb = torch.rand(D, generator=g, dtype=torch.float64) * 2 * math.pi
        phi = lambda x1, x2, W=W, bb=bb: torch.cat((math.sqrt(2.0 / D) * torch.cos((x1 * x2).double() @ W + bb), torch.ones(len(x1), 1, dtype=torch.float64)), 1)
        b = _ridge_closed(phi(*fit.p()), phi(*fit.q()), phi(*tune.p()), phi(*tune.q()))
        if best is None or b["tune_J"] > best["tune_J"]:
            best = {"fn": (lambda x1, x2, w=b["w"], c=b["c"], phi=phi: torch.tanh(c * (phi(x1, x2) @ w)).float()), "tune_J": b["tune_J"],
                    "params": {"lam": b["lam"], "c": b["c"], "bandwidth_mult": mult, "median": med, "D": D}}
    return best


def measure(fit: PairSet, tune: PairSet, ev: PairSet, *, device="cpu", mlp_steps: int = 1500, seed: int = 0) -> dict[str, Any]:
    """Fit every candidate on FIT, pick on TUNE (T = 0 included), report EVAL J of every candidate and of the pick (unclipped)."""
    out: dict[str, Any] = {"candidates": {}}
    cands = {"cosine": lambda: fit_cosine(fit, tune), "mlp": lambda: fit_mlp(fit, tune, steps=mlp_steps, seed=seed, device=device),
             "prod_ridge": lambda: fit_prod_ridge(fit, tune), "rff_ridge": lambda: fit_rff_ridge(fit, tune, seed=seed)}
    for name, fn in cands.items():
        t0 = time.time(); c = fn()
        with torch.no_grad():
            ej = j_value(c["fn"](*ev.p()), c["fn"](*ev.q()))
        out["candidates"][name] = {"tune_J": c["tune_J"], "eval_J": ej, "params": c["params"], "seconds": time.time() - t0}
    out["candidates"]["zero"] = {"tune_J": 0.0, "eval_J": 0.0, "params": {}, "seconds": 0.0}
    pick = max(out["candidates"], key=lambda k: out["candidates"][k]["tune_J"])
    out["picked"] = pick; out["picked_eval_J"] = out["candidates"][pick]["eval_J"]
    return out
