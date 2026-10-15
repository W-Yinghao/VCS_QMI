"""P158: nested critics for a joint-vs-product experiment with continuous inputs (x, y) — the joint-input counterpart of vcs_measure/nested.py
(which is left unchanged; P109 I2 / P118 depend on it).

(a) Nested critic: f_B(x_B, y_B) = f_A(m_x(x_B), m_y(y_B)) [frozen, already fitted on the coarse set] + g(x_B, y_B) [residual of the same critic
    family, output layer zero-initialised] — the fine class contains the coarse model exactly and step 0 (g = 0) is a selectable candidate.
(b) No exact product term: the inputs are continuous, so the Q expectation cannot be enumerated; fitting and every readout use the sampled
    product pairs (x_i, y_{pi(i)}) of the same draw (non-zero cyclic shift pi).

Critic families (VAL-picked, as P118 picked linear vs MLP):
    quad  — linear in the per-coordinate features [x, y, x^2, y^2, x*y] (+ bias); contains the Gaussian-pair oracle logit exactly (P158 oracle);
    mlp   — vcs_estim.critics.JointMLP (concat -> 256 -> 256 -> 1, ReLU), the E-line benchmark critic.
Fitter = the P102 / P118 settings: AdamW lr 1e-3, wd 1e-2, full batch, 300 steps, 80 / 20 FIT / VAL split, selection every 10 steps by the common
squared score J on VAL for both objectives (VCS -J and matched JS).  Nothing on EVAL is clipped, made monotone or orthogonalised.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vcs_estim.critics import JointMLP


class QuadCritic(nn.Module):
    """f(x, y) = w^T [x, y, x^2, y^2, x*y] + b  (x and y of the same width d)."""

    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(5 * d, 1)

    def forward(self, x, y):
        return self.lin(torch.cat([x, y, x * x, y * y, x * y], -1)).squeeze(-1)


class MLPCritic(nn.Module):
    def __init__(self, d: int, hidden: int = 256):
        super().__init__()
        self.net = JointMLP(d, d, hidden)

    def forward(self, x, y):
        return self.net.pairs(x, y)


def make(kind: str, d: int) -> nn.Module:
    return QuadCritic(d) if kind == "quad" else MLPCritic(d)


class NestedJointCritic(nn.Module):
    """f(x, y) = coarse(mx(x), my(y)) [frozen] + residual(x, y) [residual output layer zero-initialised]."""

    def __init__(self, coarse: nn.Module, cmap, residual: nn.Module):
        super().__init__()
        self.coarse = copy.deepcopy(coarse).eval()
        for p in self.coarse.parameters():
            p.requires_grad_(False)
        self.cmap, self.residual = cmap, residual
        last = [m for m in residual.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, x, y):
        with torch.no_grad():
            c = self.coarse(*self.cmap(x, y))
        return c + self.residual(x, y)


def _j_torch(fp, fq):
    tp, tq = torch.tanh(fp), torch.tanh(fq)
    return (tp - 0.5 * tp ** 2).mean() + (-tq - 0.5 * tq ** 2).mean()


def _js_loss(fp, fq):
    return F.softplus(-2.0 * fp).mean() + F.softplus(2.0 * fq).mean()


def fit(x, y, neg: torch.Tensor, *, objective: str, kind: str, steps: int, seed: int, coarse=None, cmap=None,
        lr: float = 1e-3, wd: float = 1e-2, val_frac: float = 0.2):
    """P102 / P118 fitter on joint inputs.  neg[i] is the partner index of unit i's Q pair (x_i, y_{neg[i]}) within the same draw."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    n = len(x); perm = torch.randperm(n, generator=g); nv = max(8, int(val_frac * n)); vi, ti = perm[:nv].to(x.device), perm[nv:].to(x.device)
    base = make(kind, x.shape[1]).to(x.device)
    m = NestedJointCritic(coarse, cmap, base).to(x.device) if coarse is not None else base
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    xt, yt, yqt = x[ti], y[ti], y[neg[ti]]; xv, yv, yqv = x[vi], y[vi], y[neg[vi]]

    def val_j():
        with torch.no_grad():
            return float(_j_torch(m(xv, yv), m(xv, yqv)))
    best, best_step, best_state = val_j(), 0, {k: v.clone() for k, v in m.state_dict().items()}
    if coarse is None:
        best = -float("inf")          # an independent fit does not offer its random initialisation as a candidate (P102 behaviour)
    for step in range(1, steps + 1):
        opt.zero_grad()
        fp, fq = m(xt, yt), m(xt, yqt)
        loss = -_j_torch(fp, fq) if objective == "vcs" else _js_loss(fp, fq)
        loss.backward(); opt.step()
        if step % 10 == 0 or step == steps:
            jv = val_j()
            if jv > best:
                best, best_step, best_state = jv, step, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval(); m.val_J = best; m.best_step = best_step
    return m


def fit_picked(x, y, neg, *, objective, steps, seed, coarse=None, cmap=None, kinds=("quad", "mlp")):
    c = {k: fit(x, y, neg, objective=objective, kind=k, steps=steps, seed=seed, coarse=coarse, cmap=cmap) for k in kinds}
    pick = max(c, key=lambda k: c[k].val_J)
    return c[pick], pick, {k: float(v.val_J) for k, v in c.items()}, int(c[pick].best_step)


@torch.no_grad()
def readouts(m, x, y, neg) -> tuple[np.ndarray, np.ndarray]:
    """Logits on the P pairs [n] and on the sampled Q pairs [n] of a draw (float64 numpy)."""
    return m(x, y).double().cpu().numpy(), m(x, y[neg]).double().cpu().numpy()
