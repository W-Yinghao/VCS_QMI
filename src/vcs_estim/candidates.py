"""Estimator package v1 (§6.1–6.2): the finite critic candidate classes.  Each returns the un-squashed output f(x, y); T = tanh f.

C0  a x'y + b                                   (two parameters; "cosine" only for unit inputs)
C1  sum_{r<=16} (u_r'x)(v_r'y) + b              (rank min(16, d), separate projections, no concat branch)
C2  concat -> 256 ReLU -> 256 ReLU -> 1         (general non-linear reference)
CQ  linear([x'y, |x|^2, |y|^2, 1])               (diagnostic: contains the Gaussian oracle l/2; identifies optimisation error)
Initialisation: C0 / CQ start at f = 0; C1 / C2 use PyTorch defaults with zero output bias.  The same rule for VCS and JS.
"""
from __future__ import annotations

import torch
from torch import nn


class C0(nn.Module):
    """a s + b with s = x'y, optionally standardised by fixed FIT statistics (mu, sd): an affine reparametrisation of the same class."""
    def __init__(self, d, a0: float = 0.0, std: tuple | None = None):
        super().__init__(); self.a = nn.Parameter(torch.tensor(float(a0))); self.b = nn.Parameter(torch.zeros(()))
        mu, sd = std if std is not None else (0.0, 1.0)
        self.register_buffer("mu", torch.tensor(float(mu))); self.register_buffer("sd", torch.tensor(float(sd)))

    def forward(self, x, y):
        return self.a * (((x * y).sum(-1) - self.mu) / self.sd) + self.b


class C1(nn.Module):
    def __init__(self, d, rank=16):
        super().__init__(); r = min(rank, d)
        self.u = nn.Linear(d, r, bias=False); self.v = nn.Linear(d, r, bias=False); self.b = nn.Parameter(torch.zeros(()))

    def forward(self, x, y):
        return (self.u(x) * self.v(y)).sum(-1) + self.b


class C2(nn.Module):
    def __init__(self, d, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], -1)).squeeze(-1)


class CQ(nn.Module):
    def __init__(self, d):
        super().__init__(); self.lin = nn.Linear(3, 1); nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, x, y):
        return self.lin(torch.stack([(x * y).sum(-1), (x * x).sum(-1), (y * y).sum(-1)], -1)).squeeze(-1)


FAMILIES = {"C0": C0, "C1": C1, "C2": C2, "CQ": CQ}


def build(family: str, d: int, c0_a0: float = 0.0, c0_std: tuple | None = None) -> nn.Module:
    """c0_a0 / c0_std: starting scale and fixed FIT standardisation of C0's inner product (defaults = the Gaussian probe: raw x'y, a = 0).
    The frozen-feature stage standardises by FIT (mean, sd): on unit vectors in a narrow cone the raw cosine has a tiny spread, so neither
    a = 0 (2 000 Adam steps cannot reach the needed scale) nor a = 5, b = 0 (tanh saturates on every pair) is readable — disclosed in P83."""
    return C0(d, c0_a0, c0_std) if family == "C0" else FAMILIES[family](d)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
