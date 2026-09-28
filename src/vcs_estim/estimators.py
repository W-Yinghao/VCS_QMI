"""Six critic-based estimators on one interface.  ``scores(critic, x, y, variant)`` returns (f_pos [n], f_neg [n, K] or [n, n-1]) for the
negative construction: 'single' (one shuffled partner), 'cyclic8' (K = 8 cyclic shifts), 'inbatch' (all n(n−1) off-diagonal pairs).
``estimate(kind, f_pos, f_neg, ...)`` returns (loss to minimise, estimate on the estimator's own scale).  All expectations are separate
means over the positive and the negative sets (the VCS form of the objective), so every estimator sees the same sample averages.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

VARIANTS = ("single", "cyclic8", "inbatch")
KINDS = ("vcs", "js", "infonce", "nwj", "dv", "smile")
TARGET = {"vcs": "S", "js": "JS2", "infonce": "MI", "nwj": "MI", "dv": "MI", "smile": "MI"}


def scores(critic, x, y, variant: str, gen: torch.Generator | None = None):
    n = len(x)
    if variant == "single":
        f_pos = critic.pairs(x, y)
        perm = torch.randperm(n, generator=gen).to(x.device)
        f_neg = critic.pairs(x, y[perm]).unsqueeze(1)
    elif variant == "cyclic8":
        f_pos = critic.pairs(x, y)
        f_neg = critic.shifted(x, y, range(1, 9))
    elif variant == "inbatch":
        M = critic.matrix(x, y); mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
        f_neg = M[mask].view(n, n - 1)
        f_pos = torch.diagonal(M)
    else:
        raise ValueError(variant)
    return f_pos, f_neg


class EMA:
    """Moving average of E_Q[exp f] for the MINE (DV) bias-corrected gradient."""
    def __init__(self, decay=0.99):
        self.decay, self.value = decay, None

    def update(self, v: torch.Tensor) -> torch.Tensor:
        v = v.detach()
        self.value = v if self.value is None else self.decay * self.value + (1 - self.decay) * v
        return self.value


def estimate(kind: str, f_pos: torch.Tensor, f_neg: torch.Tensor, *, ema: EMA | None = None, tau: float = 5.0):
    """Returns (loss, value).  f_pos: [n]; f_neg: [n, K]."""
    n, K = f_neg.shape
    if kind == "vcs":
        tp, tq = torch.tanh(f_pos), torch.tanh(f_neg)
        J = (tp - tp * tp / 2).mean() + (-tq - tq * tq / 2).mean()
        return -J, J
    if kind == "js":                                  # Deep InfoMax: E_P[−sp(−f)] − E_Q[sp(f)] (+ log 4 → 2·JSD at the optimum)
        obj = (-F.softplus(-f_pos)).mean() - F.softplus(f_neg).mean()
        return -obj, obj + math.log(4.0)
    if kind == "infonce":                             # log-softmax of the positive against its own K negatives (K = n − 1 for in-batch)
        logits = torch.cat([f_pos.unsqueeze(1), f_neg], 1)
        val = (f_pos - torch.logsumexp(logits, 1)).mean() + math.log(K + 1)
        return -val, val
    if kind == "nwj":
        val = f_pos.mean() - torch.exp(f_neg - 1.0).mean()
        return -val, val
    if kind == "dv":                                  # MINE: value = E_P f − log E_Q e^f; gradient with an EMA denominator
        eq = torch.exp(f_neg).mean()
        val = f_pos.mean() - torch.log(eq)
        if ema is not None:
            den = ema.update(eq)
            loss = -(f_pos.mean() - eq / den)          # gradient of −(E_P f − E_Q e^f / EMA): MINE's bias-corrected estimate of ∇(−value)
        else:
            loss = -val
        return loss, val
    if kind == "smile":
        eq = torch.exp(torch.clamp(f_neg, -tau, tau)).mean()
        val = f_pos.mean() - torch.log(eq)
        return -val, val
    raise ValueError(kind)


def oracle_transform(kind: str):
    """The optimal *pre-activation* critic f* as a function of the PMI (estimate() applies the estimator's own output map: VCS's tanh, so the
    VCS oracle is f* = pmi / 2, giving T* = tanh(pmi / 2) = eta; a tanh here would be applied twice — the P69-draft bug fixed 2026-09-28)."""
    if kind == "vcs":
        return lambda p: p / 2
    if kind == "nwj":
        return lambda p: 1.0 + p
    return lambda p: p                                 # js, infonce, dv, smile: f* = pmi (+ const)
