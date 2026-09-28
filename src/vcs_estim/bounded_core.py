"""Small NumPy reference for VCS-QMI fixed-output regression.

This module does not train neural networks or change the VCS objective. It
implements separately averaged P/Q moments, a bounded two-critic step, and an
exact active-face search for a SMALL (at most eight candidates) simplex QP.
All arrays are checked in float64. Do not apply the simplex enumeration to
large dictionaries.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _scores(values: ArrayLike, ndim: int) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != ndim or any(size == 0 for size in arr.shape):
        raise ValueError(f"Expected a nonempty {ndim}-D array, got {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError("Scores must be finite.")
    if np.max(np.abs(arr)) > 1.0 + 1e-12:
        raise ValueError("This reference accepts bounded critic scores only.")
    return arr


def j_from_scores(positive: ArrayLike, negative: ArrayLike) -> float:
    """Raw J: positive and negative distributions are averaged separately."""
    p, q = _scores(positive, 1), _scores(negative, 1)
    return float(np.mean(p - 0.5 * p * p) + np.mean(-q - 0.5 * q * q))


def balanced_squared_risk(positive: ArrayLike, negative: ArrayLike) -> float:
    p, q = _scores(positive, 1), _scores(negative, 1)
    return float(0.5 * np.mean((1 - p) ** 2) + 0.5 * np.mean((-1 - q) ** 2))


@dataclass(frozen=True)
class ResidualStep:
    coefficient: float
    A: float
    B: float
    predicted_gain: float
    zero_direction: bool


def residual_step(
    positive_base: ArrayLike,
    negative_base: ArrayLike,
    positive_candidate: ArrayLike,
    negative_candidate: ArrayLike,
    *,
    zero_tolerance: float = 1e-14,
) -> ResidualStep:
    """Fit lambda on a designated TUNE set, never on final EVAL scores."""
    p0, q0 = _scores(positive_base, 1), _scores(negative_base, 1)
    p1, q1 = _scores(positive_candidate, 1), _scores(negative_candidate, 1)
    if p0.shape != p1.shape or q0.shape != q1.shape:
        raise ValueError("Corresponding base/candidate scores must use identical pairs.")
    if zero_tolerance < 0:
        raise ValueError("zero_tolerance must be nonnegative.")
    dp, dq = p1 - p0, q1 - q0
    A = float(0.5 * np.mean((1-p0)*dp) + 0.5 * np.mean((-1-q0)*dq))
    B = float(0.5 * np.mean(dp*dp) + 0.5 * np.mean(dq*dq))
    if B <= zero_tolerance:
        return ResidualStep(0.0, A, B, 0.0, True)
    coefficient = float(np.clip(A/B, 0.0, 1.0))
    return ResidualStep(coefficient, A, B,
                        2*coefficient*A - coefficient*coefficient*B, False)


def dictionary_moments(positive: ArrayLike, negative: ArrayLike) -> tuple[FloatArray, FloatArray]:
    p, q = _scores(positive, 2), _scores(negative, 2)
    if p.shape[1] != q.shape[1]:
        raise ValueError("Positive and negative dictionaries must have equal column counts.")
    d = p.mean(axis=0) - q.mean(axis=0)
    G = 0.5*(p.T @ p/len(p) + q.T @ q/len(q))
    return d, (G+G.T)/2


@dataclass(frozen=True)
class SimplexFit:
    weights: FloatArray
    objective: float
    kkt_gap: float
    candidates_considered: int


def fit_small_simplex(positive: ArrayLike, negative: ArrayLike, *, tolerance: float = 1e-10) -> SimplexFit:
    """Maximise d.w - w.G.w over the simplex (m <= 8).

    Enumerates active faces and solves their equality-constrained stationarity
    systems. Degenerate systems use least squares. Tiny negative WEIGHTS from
    round-off are projected onto the face; critic scores are never clipped.
    """
    d, G = dictionary_moments(positive, negative)
    m = len(d)
    if not 1 <= m <= 8:
        raise ValueError("Active-face reference is limited to 1..8 critics.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    best_w, best_value, considered = None, -np.inf, 0
    for mask in range(1, 1 << m):
        ix = np.asarray([j for j in range(m) if mask & (1 << j)], dtype=int)
        k = len(ix)
        K = np.zeros((k+1, k+1), dtype=np.float64)
        K[:k, :k] = 2*G[np.ix_(ix, ix)]
        K[:k, k] = 1.0
        K[k, :k] = 1.0
        rhs = np.r_[d[ix], 1.0]
        sol = np.linalg.lstsq(K, rhs, rcond=1e-12)[0]
        residual = np.linalg.norm(K @ sol-rhs, ord=np.inf)
        if residual > 100*tolerance or np.min(sol[:k]) < -tolerance:
            continue
        w = np.zeros(m, dtype=np.float64)
        w[ix] = np.maximum(sol[:k], 0.0)
        if w.sum() <= 0:
            continue
        w /= w.sum()
        considered += 1
        value = float(d @ w - w @ G @ w)
        if value > best_value + 1e-14:
            best_w, best_value = w, value
    if best_w is None:
        raise ArithmeticError("No feasible face found; inspect scores and conditioning.")
    gradient = d-2*G @ best_w
    gap = float(np.max(gradient)-best_w @ gradient)
    if gap > max(1e-8, 100*tolerance):
        raise ArithmeticError(f"Simplex solution fails KKT check: gap={gap}.")
    best_w.setflags(write=False)
    return SimplexFit(best_w, best_value, max(0.0, gap), considered)
