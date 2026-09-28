"""One critic class for every estimator: a joint MLP on concat(x, y) → 256 → 256 → 1 (ReLU).  Pair scores are produced for (i) aligned
pairs, (ii) K shifted partners per anchor, or (iii) the full in-batch matrix, all through the same network."""
from __future__ import annotations

import torch
from torch import nn


class JointMLP(nn.Module):
    def __init__(self, dx: int, dy: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dx + dy, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def pairs(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:          # [n, dx], [n, dy] -> [n]
        return self.net(torch.cat([x, y], -1)).squeeze(-1)

    def shifted(self, x: torch.Tensor, y: torch.Tensor, shifts) -> torch.Tensor:   # -> [n, K]
        return torch.stack([self.pairs(x, torch.roll(y, s, 0)) for s in shifts], 1)

    def matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:         # -> [n, n], entry (i, j) = f(x_i, y_j)
        n = len(x); xi = x.repeat_interleave(n, 0); yj = y.repeat(n, 1)
        return self.pairs(xi, yj).view(n, n)


class OracleCritic:
    """f(x, y) = g(pmi(x, y)) for a known PMI function; offers the same three score interfaces."""
    def __init__(self, pmi_fn, transform=lambda p: p):
        self.pmi_fn, self.tf = pmi_fn, transform

    def pairs(self, x, y):
        return self.tf(self.pmi_fn(x, y))

    def shifted(self, x, y, shifts):
        return torch.stack([self.pairs(x, torch.roll(y, s, 0)) for s in shifts], 1)

    def matrix(self, x, y):
        n = len(x); xi = x.repeat_interleave(n, 0); yj = y.repeat(n, 1)
        return self.pairs(xi, yj).view(n, n)
