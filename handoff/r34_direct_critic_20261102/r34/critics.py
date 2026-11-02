"""R34 critics and objectives. g(x,y) is the real log-ratio-type score; T = tanh(g/2).
D: g = a*cos(Wx hx, Wy hy) + b with a = 2*s, b = -2*s*o, s = softplus(raw)+1e-6 (R32 F1 AffineCosineCritic
   parametrisation; init s=2, o=0.5 -> a=4, b=-2), projection 256 (linear with bias per modality).
X: g = MLP([hx, hy]) 2048 -> 256 -> GELU -> 256 -> GELU -> 1; first layer evaluated blockwise
   Wx hx + Wy hy + b1 (mathematically equal to the concat Linear; checked in tests/gate).
Objectives on a BxB in-batch matrix of IID-with-replacement record draws (R32 F1 rule):
   P = record diagonal (weight 1/B), Q = off-diagonal records (weight 1/(B(B-1))), same-image Q retained.
   J  = E_P[T - T^2/2] - E_Q[T + T^2/2];  loss = -J
   BCE = E_P softplus(-g) + E_Q softplus(g)
   NCE = 0.5*(CE(rows, diag) + CE(cols, diag)), logits = g (diagonal positives, all other draws candidates)
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

STRUCTURES = ('D', 'X')
OBJECTIVES = ('J', 'BCE', 'NCE')


class DualCosine(nn.Module):
    def __init__(self, dx=1024, dy=1024, proj=256, scale=2.0, offset=0.5):
        super().__init__()
        self.image_head = nn.Linear(dx, proj)
        self.text_head = nn.Linear(dy, proj)
        self.scale_floor = 1e-6
        self.raw_scale = nn.Parameter(torch.tensor(math.log(math.expm1(scale - self.scale_floor))))
        self.offset = nn.Parameter(torch.tensor(float(offset)))

    def ab(self):
        s = F.softplus(self.raw_scale) + self.scale_floor
        return 2 * s, -2 * s * self.offset

    def embed(self, x, y):
        return F.normalize(self.image_head(x), dim=-1), F.normalize(self.text_head(y), dim=-1)

    def pair_scores(self, x, y):
        """all image x caption pairs: [n_x, n_y]"""
        u, v = self.embed(x, y)
        a, b = self.ab()
        return a * (u @ v.T) + b


class JointMLP(nn.Module):
    def __init__(self, dx=1024, dy=1024, hidden=256):
        super().__init__()
        self.dx = dx
        self.l1 = nn.Linear(dx + dy, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

    def first(self, x, y):
        w = self.l1.weight
        return x @ w[:, :self.dx].T + self.l1.bias, y @ w[:, self.dx:].T

    def head(self, h1):
        return self.l3(F.gelu(self.l2(F.gelu(h1)))).squeeze(-1)

    def pair_scores(self, x, y, chunk=None):
        ax, ay = self.first(x, y)
        if chunk is None:
            return self.head(ax[:, None, :] + ay[None, :, :])
        return torch.cat([self.head(ax[i:i + chunk, None, :] + ay[None, :, :]) for i in range(0, len(ax), chunk)], 0)

    def concat_scores(self, x, y):
        """explicit concat Linear (reference for the blockwise identity)"""
        n, m = len(x), len(y)
        z = torch.cat([x[:, None, :].expand(n, m, x.shape[1]), y[None, :, :].expand(n, m, y.shape[1])], -1)
        return self.head(self.l1(z))


def build(structure):
    return DualCosine() if structure == 'D' else JointMLP()


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pq_weights(b, device, dtype=torch.float32):
    eye = torch.eye(b, device=device, dtype=dtype)
    return eye / b, (1 - eye) / (b * (b - 1))


def j_value(g, p, q):
    t = torch.tanh(g / 2)
    return (p * (t - .5 * t * t)).sum() - (q * (t + .5 * t * t)).sum()


def bce_value(g, p, q):
    return (p * F.softplus(-g)).sum() + (q * F.softplus(g)).sum()


def nce_value(g):
    lab = torch.arange(g.shape[0], device=g.device)
    return .5 * (F.cross_entropy(g, lab) + F.cross_entropy(g.T, lab))


def loss_of(objective, g, p, q):
    if objective == 'J':
        return -j_value(g, p, q)
    if objective == 'BCE':
        return bce_value(g, p, q)
    if objective == 'NCE':
        return nce_value(g)
    raise ValueError(objective)
