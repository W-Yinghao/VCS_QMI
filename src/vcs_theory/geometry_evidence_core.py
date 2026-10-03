"""Small reference core for VCS v6. No repository or dataset dependencies.

Copied verbatim (code unchanged; only this attribution paragraph added) from the v6 package VCS_Server_Tasks_and_Theory_v6_20261003.zip,
support/geometry_evidence_core.py (v6 research-implementation materials, 2026-10-03), for P121 / V6-THEORY.

This module checks mathematical objects; it is NOT a replacement SSL trainer.
The optional PyTorch scorer demonstrates fixed-buffer / matrix API consistency.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any
import numpy as np


def distributions(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("p and q must have equal shapes and finite values")
    if (p < 0).any() or (q < 0).any() or not np.isclose(p.sum(), 1) or not np.isclose(q.sum(), 1):
        raise ValueError("p and q must be probability mass functions")
    return p, q


def posterior(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    p, q = distributions(p, q)
    m = (p + q) / 2
    eta = np.divide(p - q, p + q, out=np.zeros_like(p), where=(p + q) > 0)
    return eta, m, float(np.sum(m * eta**2))


def j_population(p: np.ndarray, q: np.ndarray, t: np.ndarray) -> float:
    p, q = distributions(p, q)
    t = np.asarray(t, dtype=np.float64)
    if t.shape != p.shape or not np.isfinite(t).all():
        raise ValueError("t must match the distribution shape and be finite")
    return float(np.sum(p * (t - t**2 / 2)) + np.sum(q * (-t - t**2 / 2)))


def conditional_mean(values: np.ndarray, weights: np.ndarray, labels: np.ndarray) -> np.ndarray:
    v, w, labels = np.asarray(values).ravel(), np.asarray(weights).ravel(), np.asarray(labels).ravel()
    if v.shape != w.shape or v.shape != labels.shape or (w < 0).any():
        raise ValueError("values, weights, labels must align; weights must be non-negative")
    out = np.empty_like(v, dtype=np.float64)
    for group in np.unique(labels):
        idx = labels == group
        mass = w[idx].sum()
        out[idx] = np.sum(v[idx] * w[idx]) / mass if mass > 0 else 0
    return out.reshape(np.asarray(values).shape)


def projection_decomposition(p, q, z_labels, s_labels, t) -> dict[str, float]:
    eta, m, target = posterior(p, q)
    zl, sl = np.asarray(z_labels).ravel(), np.asarray(s_labels).ravel()
    for z in np.unique(zl):
        if len(np.unique(sl[zl == z])) != 1:
            raise ValueError("s must be a deterministic function of Z labels")
    ez = conditional_mean(eta, m, z_labels)
    es = conditional_mean(eta, m, s_labels)
    t = np.asarray(t, dtype=np.float64)
    for s in np.unique(sl):
        if np.ptp(t.ravel()[sl == s]) > 1e-12:
            raise ValueError("t must depend only on s")
    terms = [float(np.sum(m * a**2)) for a in (eta - ez, ez - es, es - t)]
    gap = target - j_population(p, q, t)
    return dict(S=target, gap=gap, representation=terms[0], similarity=terms[1],
                calibration=terms[2], residual=gap - sum(terms))


def latent_model(n_states: int = 12, n_classes: int = 4, seed: int = 73) -> dict[str, np.ndarray | float]:
    if min(n_states, n_classes) < 2:
        raise ValueError("Need at least two states and classes")
    rng = np.random.default_rng(seed)
    pi = rng.dirichlet(np.ones(n_classes))
    a = rng.uniform(.02, 1.0, (n_states, n_classes))
    a /= a.sum(axis=0, keepdims=True)
    marginal = a @ pi
    p = (a * pi) @ a.T
    q = np.outer(marginal, marginal)
    py_u = (a * pi) / marginal[:, None]
    eta, m, target = posterior(p, q)
    return dict(P=p, Q=q, M=m, eta=eta, S=target, marginal=marginal,
                emission=a, prior=pi, class_posterior=py_u)


def push_joint(p: np.ndarray, left_map: np.ndarray, right_map: np.ndarray | None = None):
    p = np.asarray(p, dtype=np.float64)
    lm = np.asarray(left_map, dtype=int)
    rm = lm if right_map is None else np.asarray(right_map, dtype=int)
    if p.shape != (len(lm), len(rm)) or min(lm.min(), rm.min()) < 0:
        raise ValueError("Invalid state map")
    out = np.zeros((int(lm.max()) + 1, int(rm.max()) + 1))
    np.add.at(out, (np.broadcast_to(lm[:, None], p.shape), np.broadcast_to(rm[None, :], p.shape)), p)
    return out


def curve_logits(s, a: float = 2., kappa: float = .5, curvature: float = 0.):
    if not np.isfinite([a, kappa, curvature]).all() or a <= 0 or abs(curvature) > .25:
        raise ValueError("Require a > 0, finite values, |curvature| <= .25")
    s = np.asarray(s, dtype=np.float64)
    if (np.abs(s) > 1 + 1e-10).any():
        raise ValueError("s must be a cosine in [-1,1]")
    return a * s - a * kappa + a * curvature * s * (1 - s)


def curve_slope(s, a: float = 2., curvature: float = 0.):
    return a * (1 + curvature * (1 - 2 * np.asarray(s)))


def contribution(c, f):
    t = np.tanh(f)
    return c * t - .5 * t**2


def contribution_gradient(c, s, a=2., kappa=.5, curvature=0.):
    f = curve_logits(s, a, kappa, curvature)
    t = np.tanh(f)
    return (c - t) * (1 - t**2) * curve_slope(s, a, curvature)


def actual_zero(a=2., kappa=.5, curvature=0.):
    lo, hi = -1., 1.
    if curve_logits(lo, a, kappa, curvature) * curve_logits(hi, a, kappa, curvature) > 0:
        return None
    for _ in range(64):
        mid = (lo + hi) / 2
        if curve_logits(mid, a, kappa, curvature) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def pair_masks(base_ids: np.ndarray):
    ids = np.asarray(base_ids)
    if ids.ndim != 1:
        raise ValueError("base_ids must be one-dimensional")
    same = ids[:, None] == ids[None, :]
    positive = same & ~np.eye(len(ids), dtype=bool)
    negative = ~same
    return positive, negative


def gram_diagnostics(g: np.ndarray) -> dict[str, float]:
    g = np.asarray(g, dtype=float)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError("g must be square")
    vals = np.linalg.eigvalsh((g + g.T) / 2)
    return dict(symmetry_error=float(np.max(np.abs(g - g.T))),
                diagonal_error=float(np.max(np.abs(np.diag(g) - 1))),
                negative_eigen_mass=float(-vals[vals < 0].sum()),
                range_excess=float(np.maximum(np.abs(g) - 1, 0).max()))

try:
    import torch
    from torch import nn
except ImportError:  # NumPy-only checks remain available.
    torch = None
    nn = None

if nn is not None:
    class FixedAnchoredCurveCritic(nn.Module):
        """Fixed logits a*s+b+a*lambda*s*(1-s); lambda=0 equals fixed cosine.

        Inputs must already be row-L2-normalized; this class does not silently
        renormalize embeddings or alter training pair weights.
        """
        trainable_affine = False
        def __init__(self, feature_dim: int, scale: float = 2., kappa: float = .5, curvature: float = 0.):
            super().__init__()
            if feature_dim < 1 or not np.isfinite([scale, kappa, curvature]).all() or scale <= 0 or abs(curvature) > .25:
                raise ValueError("Invalid scorer parameters")
            self.feature_dim = feature_dim
            self.register_buffer("scale", torch.tensor(float(scale)))
            self.register_buffer("bias", torch.tensor(float(-scale * kappa)))
            self.register_buffer("curvature", torch.tensor(float(curvature)))
        def logits_from_similarity(self, s):
            return self.scale * s + self.bias + self.scale * self.curvature * s * (1 - s)
        def logits(self, left, right):
            if left.ndim != 2 or left.shape != right.shape or left.shape[1] != self.feature_dim:
                raise ValueError("Expected equal [N, feature_dim] pairs")
            return self.logits_from_similarity((left * right).sum(-1))
        def forward(self, left, right):
            return torch.tanh(self.logits(left, right))
        def score_matrix(self, s):
            return torch.tanh(self.logits_from_similarity(s))
        def embed(self, z):
            return z
        def derivative_from_similarity(self, s):
            t = self.score_matrix(s)
            return (1 - t.square()) * self.scale * (1 + self.curvature * (1 - 2 * s))
