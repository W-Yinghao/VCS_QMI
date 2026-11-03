"""R37 objectives (R36 objectives.py + loss_list_masked for N_PRESENT_ONLY). R36 objectives (R35 objectives.py unchanged + list_rows_pq for the M-SHIELD oracle diagnostic). R35: T = tanh(g/2); J = E_P[T - T^2/2] - E_Q[T + T^2/2] (unchanged theory).
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


def list_rows_pq(g, a, q, valid, objective):
    """R36 M-SHIELD_ORACLE rows with explicit reference q (renormalised on valid slots) and NCE support mask valid.
    J = sum a(T-.5T^2) - sum q(T+.5T^2); BCE = sum a softplus(-g) + sum q softplus(g); NCE = -sum a log_softmax over
    valid slots only (masked slots excluded from the denominator; 0*(-inf) avoided). a must be 0 off valid."""
    if objective == 'J':
        t = torch.tanh(g / 2)
        return (a * (t - .5 * t * t)).sum(1) - (q * (t + .5 * t * t)).sum(1)
    if objective == 'BCE':
        return (a * F.softplus(-g)).sum(1) + (q * F.softplus(g)).sum(1)
    if objective == 'NCE':
        lg = F.log_softmax(g.masked_fill(~valid, float('-inf')), dim=1).masked_fill(~valid, 0.0)
        return -(a * lg).sum(1)
    raise ValueError(objective)


def loss_list_pq(objective, g, a, q, valid):
    v = list_rows_pq(g, a, q, valid, objective).mean()
    return -v if objective == 'J' else v


def j_parts(g, a, slot_mask):
    """per-row J restricted to slots in slot_mask (uniform reference 1/K over ALL K slots, i.e. the additive part of
    the original list J): parts over real slots and over the null slot add up to the total J."""
    t = torch.tanh(g / 2); K = g.shape[1]
    return ((a * (t - .5 * t * t) - (t + .5 * t * t) / K) * slot_mask).sum(1)


def loss_list_masked(objective, g, a, mask, p_d):
    """R37 N_PRESENT_ONLY: sum(mask * row_value) / (n_rows * p_d) (rows with mask 0 contribute 0, not redrawn); p_d is the
    fixed training-pool presence probability of the direction, so E[weight] per direction is the same as the unmasked
    mean. Equals loss_list when mask is all ones and p_d = 1."""
    v = (list_rows(g, a, objective) * mask).sum() / (g.shape[0] * p_d)
    return -v if objective == 'J' else v
