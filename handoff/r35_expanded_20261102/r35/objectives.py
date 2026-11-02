"""R35 objectives. T = tanh(g/2); J = E_P[T - T^2/2] - E_Q[T + T^2/2] (unchanged theory).
A/B (records, R34 rule): in-batch BxB matrix of IID record draws, P = diagonal 1/B, Q = off-diagonal 1/(B(B-1)),
    same-image product pairs retained; NCE = bidirectional CE with diagonal labels.
C/D (candidate lists): row = (query, 32-slot list), A = observed-positive indicator normalised per row (P_C),
    Q_C = P_Z x Uniform(32) (positive slots included). J_row = sum A(T-.5T^2) - mean(T+.5T^2);
    BCE_row = sum A softplus(-g) + mean softplus(g); NCE_row = -sum A log_softmax(g). No softmax before J and
    no R29 (K*pi-1)/(K*pi+1) link. Loss = mean over rows of the two directions, averaged (0.5 each)."""
from __future__ import annotations
import torch
import torch.nn.functional as F

OBJECTIVES = ('J', 'BCE', 'NCE')


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


def loss_records(objective, g, p, q):
    if objective == 'J':
        return -j_value(g, p, q)
    if objective == 'BCE':
        return bce_value(g, p, q)
    if objective == 'NCE':
        return nce_value(g)
    raise ValueError(objective)


def list_rows(g, a, objective):
    """per-row value: J (to maximise) / BCE / NCE (to minimise). g,a [rows, slots]."""
    if objective == 'J':
        t = torch.tanh(g / 2)
        return (a * (t - .5 * t * t)).sum(1) - (t + .5 * t * t).mean(1)
    if objective == 'BCE':
        return (a * F.softplus(-g)).sum(1) + F.softplus(g).mean(1)
    if objective == 'NCE':
        return -(a * F.log_softmax(g, dim=1)).sum(1)
    raise ValueError(objective)


def loss_list(objective, g, a):
    v = list_rows(g, a, objective).mean()
    return -v if objective == 'J' else v
