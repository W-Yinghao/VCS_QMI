"""I2 pilot (package v4 §I2): two changes to the P102 finite-critic increments, evaluated without forcing the theory.

(a) Nested critic: the fine-set critic is f_B(z_B, n, y) = f_A(m(z_B), n, y) + g(z_B, n, y), where f_A is the already-fitted coarse critic (frozen),
    m the deterministic map from the fine to the coarse information set, and g a residual of the T1 linear / MLP class whose output layer starts at
    zero — so the fine function class contains the coarse model exactly, and step 0 (g = 0) is a selectable candidate.
(b) Exact product term: with binary N and known P(N = 1 | Y), the Q expectation of any g(T(z, n', y)) is Σ_{n'∈{0,1}} P(n' | y) g(T(z, n', y)) on the
    observed (z, y) — no sampled within-class POOL partner.  Used for fitting (both objectives) and/or for the readout.

Readouts work on a generic Q representation: T_Q [n, m] with row weights w [n, m] summing to 1 (m = 1 sampled, m = 2 enumerated).
Nothing on EVAL is clipped, made monotone or orthogonalised.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vcs_estim.increments import YPairLinear, YPairMLP


# ---------------------------------------------------------------------------------------------------------------------- weighted readouts
def j_w(tp: np.ndarray, tq: np.ndarray, wq: np.ndarray) -> float:
    return float(np.mean(tp - 0.5 * tp ** 2) + np.mean(np.sum(wq * (-tq - 0.5 * tq ** 2), axis=1)))


def em_w(gp: np.ndarray, gq: np.ndarray, wq: np.ndarray) -> float:
    return float(0.5 * np.mean(gp) + 0.5 * np.mean(np.sum(wq * gq, axis=1)))


def increment_w(B: tuple, A: tuple, wq: np.ndarray) -> dict:
    (bp, bq), (ap_, aq) = B, A
    jb, ja = j_w(bp, bq, wq), j_w(ap_, aq, wq); dt = em_w((bp - ap_) ** 2, (bq - aq) ** 2, wq)
    return {"J_B": jb, "J_A": ja, "delta_J": jb - ja, "D_T": dt, "r_BA": (jb - ja) - dt}


def chain_w(chain: list, names: list, wq: np.ndarray) -> dict:
    steps = {f"{names[i]}->{names[i + 1]}": increment_w(chain[i], chain[i + 1], wq) for i in range(len(chain) - 1)}
    total = em_w((chain[0][0] - chain[-1][0]) ** 2, (chain[0][1] - chain[-1][1]) ** 2, wq)
    J = {nm: j_w(c[0], c[1], wq) for nm, c in zip(names, chain)}
    return {"J": J, "steps": steps, "R_orth": float(sum(s["D_T"] for s in steps.values()) - total),
            "nesting_violations": int(sum(J[names[i]] < J[names[i + 1]] for i in range(len(names) - 1)))}


# ---------------------------------------------------------------------------------------------------------------------- critics
class NestedCritic(nn.Module):
    """f(z, n, y) = coarse(map(z), n, y) [frozen] + residual(z, n, y) [residual output layer zero-initialised]."""

    def __init__(self, coarse: nn.Module, cmap, residual: nn.Module):
        super().__init__()
        self.coarse = copy.deepcopy(coarse).eval()
        for p in self.coarse.parameters():
            p.requires_grad_(False)
        self.cmap, self.residual = cmap, residual
        last = [m for m in residual.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, z, n, y):
        with torch.no_grad():
            c = self.coarse(self.cmap(z), n, y)
        return c + self.residual(z, n, y)


def _q_logits(m, z, y, neg: dict):
    """Q logits [n, m] and weights [n, m]: neg = {'n_neg': tensor} (sampled) or {'p1': tensor} (exact enumeration with P(N = 1 | y))."""
    if "n_neg" in neg:
        return m(z, neg["n_neg"], y)[:, None], torch.ones(len(z), 1, device=z.device)
    p1 = neg["p1"]
    f0 = m(z, torch.zeros_like(y), y); f1 = m(z, torch.ones_like(y), y)
    return torch.stack((f0, f1), 1), torch.stack((1 - p1, p1), 1)


def _j_torch(fp, fq, wq):
    tp, tq = torch.tanh(fp), torch.tanh(fq)
    return (tp - 0.5 * tp ** 2).mean() + (wq * (-tq - 0.5 * tq ** 2)).sum(1).mean()


def _js_loss(fp, fq, wq):
    return F.softplus(-2.0 * fp).mean() + (wq * F.softplus(2.0 * fq)).sum(1).mean()


def _sub(neg: dict, idx):
    return {k: v[idx] for k, v in neg.items()}


def fit(z, n, y, neg: dict, *, objective: str, kind: str, n_classes: int, steps: int, seed: int, coarse=None, cmap=None,
        lr: float = 1e-3, wd: float = 1e-2, val_frac: float = 0.2):
    """P102 fitter (AdamW, full batch, selection every 10 steps by the common squared score on the 20 % VAL split), with optional nesting
    (residual on top of a frozen coarse critic; step 0 is evaluated and selectable) and either negative representation."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(z), generator=g); nv = max(8, int(val_frac * len(z))); vi, ti = perm[:nv].to(z.device), perm[nv:].to(z.device)
    base = (YPairLinear(z.shape[1], n_classes) if kind == "linear" else YPairMLP(z.shape[1], n_classes)).to(z.device)
    m = NestedCritic(coarse, cmap, base).to(z.device) if coarse is not None else base
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    negt, negv = _sub(neg, ti), _sub(neg, vi)

    def val_j():
        with torch.no_grad():
            fq, wq = _q_logits(m, z[vi], y[vi], negv); return float(_j_torch(m(z[vi], n[vi], y[vi]), fq, wq))
    best, best_step, best_state = val_j(), 0, {k: v.clone() for k, v in m.state_dict().items()}
    if coarse is None:
        best = -float("inf")          # an independent fit does not offer its random initialisation as a candidate (P102 behaviour)
    for step in range(1, steps + 1):
        opt.zero_grad()
        fp = m(z[ti], n[ti], y[ti]); fq, wq = _q_logits(m, z[ti], y[ti], negt)
        loss = -_j_torch(fp, fq, wq) if objective == "vcs" else _js_loss(fp, fq, wq)
        loss.backward(); opt.step()
        if step % 10 == 0 or step == steps:
            jv = val_j()
            if jv > best:
                best, best_step, best_state = jv, step, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval(); m.val_J = best; m.best_step = best_step
    return m


def fit_picked(z, n, y, neg, *, objective, n_classes, steps, seed, coarse=None, cmap=None):
    c = {k: fit(z, n, y, neg, objective=objective, kind=k, n_classes=n_classes, steps=steps, seed=seed, coarse=coarse, cmap=cmap) for k in ("linear", "mlp")}
    pick = max(c, key=lambda k: c[k].val_J)
    return c[pick], pick, {k: float(v.val_J) for k, v in c.items()}, int(c[pick].best_step)


@torch.no_grad()
def readouts(m, z, n, y, neg_sampled: dict, p1: torch.Tensor):
    """T on P [n], T on Q sampled [n, 1] and T on Q enumerated [n, 2] with their weights."""
    tp = torch.tanh(m(z, n, y)).double().cpu().numpy()
    fqs, wqs = _q_logits(m, z, y, neg_sampled); fqe, wqe = _q_logits(m, z, y, {"p1": p1})
    return tp, (torch.tanh(fqs).double().cpu().numpy(), wqs.double().cpu().numpy()), (torch.tanh(fqe).double().cpu().numpy(), wqe.double().cpu().numpy())
