"""Gaussian product-kernel RuLSIF with complete within-scene product moments.

No random Fourier features or sampled permutation approximation. Natural images
use V; the same algebra exposes U for the separately registered T03 follow-up.
g = sum_j theta_j k_x(x,cx_j) k_y(y,cy_j), T = g - 1 is NOT clipped to [-1,1].
The unconstrained ridge solution follows the paper's equations (4)-(6): neither
coefficients nor g/T are clipped. Negative estimates remain visible in diagnostics.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def gaussian_features(values, centres, bandwidth):
    if values.ndim != 2 or centres.ndim != 2 or values.shape[1] != centres.shape[1]:
        raise ValueError("Kernel observations and centres must have matching feature widths")
    if not math.isfinite(float(bandwidth)) or float(bandwidth) <= 0:
        raise ValueError("Gaussian bandwidth must be finite and positive")
    distance = (values.square().sum(-1, keepdim=True) + centres.square().sum(-1)[None]
                - 2 * values @ centres.T).clamp_min(0)
    return torch.exp(-distance / (2 * bandwidth**2))


def kernel_moments(left, right, q_mode="v"):
    """hP, HP, hQ, HQ for aligned features [N,M], with no N x N pair materialisation."""
    if left.ndim != 2 or left.shape != right.shape or left.shape[0] < 1:
        raise ValueError("Paired kernel features must have equal nonempty [N,M] shapes")
    n = left.shape[0]
    positive = left * right
    hp = positive.mean(0)
    hp2_sum = positive.T @ positive
    hq_sum = left.sum(0) * right.sum(0)
    hq2_sum = (left.T @ left) * (right.T @ right)
    if q_mode == "v":
        denominator = n * n
    elif q_mode == "u" and n > 1:
        hq_sum = hq_sum - positive.sum(0)
        hq2_sum = hq2_sum - hp2_sum
        denominator = n * (n - 1)
    else:
        raise ValueError("q_mode must be v, or u with at least two observations")
    return hp, hp2_sum / n, hq_sum / denominator, hq2_sum / denominator


class ExactRuLSIF(nn.Module):
    def __init__(self, centres_left, centres_right, bandwidth, ridge, alpha=0.5):
        super().__init__()
        if centres_left.ndim != 2 or centres_right.ndim != 2:
            raise ValueError("Kernel centres must be matrices")
        if len(centres_left) != len(centres_right) or len(centres_left) == 0:
            raise ValueError("Kernel centres must be paired and nonempty")
        if alpha != 0.5:
            raise ValueError("The V10 common-J relation requires alpha = 1/2")
        if not math.isfinite(ridge) or ridge <= 0 or not math.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("Positive finite ridge and bandwidth required")
        self.register_buffer("centres_left", centres_left.detach().clone())
        self.register_buffer("centres_right", centres_right.detach().clone())
        self.register_buffer("coefficients", centres_left.new_zeros(len(centres_left)))
        self.register_buffer("fitted", torch.tensor(False, device=centres_left.device))
        self.bandwidth, self.ridge, self.alpha = float(bandwidth), float(ridge), alpha

    def features(self, left, right):
        return (gaussian_features(left, self.centres_left, self.bandwidth),
                gaussian_features(right, self.centres_right, self.bandwidth))

    @torch.no_grad()
    def fit(self, scenes, *, q_mode="v"):
        """Each scene has equal weight; accumulation/linear solve in float64."""
        hp = h = None
        count = 0
        for left, right in scenes:
            a, b = self.features(left, right)
            p, p2, _, q2 = kernel_moments(a.double(), b.double(), q_mode)
            local_h = self.alpha * p2 + (1 - self.alpha) * q2
            hp = p.clone() if hp is None else hp + p
            h = local_h.clone() if h is None else h + local_h
            count += 1
        if count == 0:
            raise ValueError("No FIT scenes for RuLSIF")
        hp, h = hp / count, h / count
        system = h + self.ridge * torch.eye(len(hp), device=h.device, dtype=h.dtype)
        raw = torch.linalg.solve(system, hp)
        if not torch.isfinite(raw).all():
            raise RuntimeError("Non-finite RuLSIF coefficients")
        self.coefficients.copy_(raw.to(self.coefficients))
        self.fitted.fill_(True)
        return {"scene_count": count, "q_mode": q_mode, "alpha": self.alpha,
                "linear_solve_dtype": "float64", "negative_coefficients": int((raw < 0).sum()),
                "linear_solve_residual": float(torch.linalg.vector_norm(system @ raw - hp)),
                "coefficient_postprocessing": "none; unconstrained RuLSIF; no pointwise clipping of g"}

    def common_j(self, left, right, *, q_mode="v"):
        if not bool(self.fitted):
            raise RuntimeError("RuLSIF has not been fitted")
        a, b = self.features(left, right)
        hp, hp2, hq, hq2 = kernel_moments(a, b, q_mode)
        w = self.coefficients
        gp, gq = hp @ w, hq @ w
        gp2, gq2 = w @ hp2 @ w, w @ hq2 @ w
        # Expanded J(g-1); gq cancels but is retained in the explicit moments.
        tp, tq = gp - 1, gq - 1
        tp2, tq2 = gp2 - 2 * gp + 1, gq2 - 2 * gq + 1
        return tp - tq - 0.5 * (tp2 + tq2)

    def forward(self, left, right):
        if not bool(self.fitted):
            raise RuntimeError("RuLSIF has not been fitted")
        a, b = self.features(left, right)
        return (a * b) @ self.coefficients
