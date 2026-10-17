"""V10 T03 objectives in U mode (decision 9.1): exact U moments (package backend), uniform-derangement perm1 / perm8, exact table
statistics from cell counts.

J(T) = E_P T - E_Q T - (E_P T^2 + E_Q T^2) / 2 with the empirical U reference
    Q_U = sum_{i != j} delta_(x_i, y_j) / (n (n - 1)).
MF_VCS_M0 / M1: package objective_terms / QuadraticHead on SceneBatch(q_mode='u') (complete U moments; one scene, weight 1).
perm1 / perm8: negatives (x_i, y_sigma(i)) for R independent UNIFORM DERANGEMENTS sigma (rejection of uniform permutations with a
fixed point: conditional on acceptance a uniform permutation is a uniform derangement).  For every i and j != i a uniform derangement
has P(sigma(i) = j) = 1 / (n - 1), so the conditional mean of the perm objective (value and gradient: both are linear in the per-pair
terms) is the complete U objective; plain randperm averages to V, a fixed cyclic shift is a single deterministic derangement, and
"repairing" fixed points changes the law - none is used.
Complete empirical statistics of a frozen critic on discrete data are exact functions of the m x m cell table T and the counts N:
    p_hat = N_xy / n,  q_hat_U = (N_x N_y - N_xy) / (n (n - 1)),  q_hat_V = N_x N_y / n^2,
    J_hat = sum_xy [ p_hat (T - T^2/2) - q_hat (T + T^2/2) ]                                         (negatives kept, never clipped)
(used for CAL selection and every readout; training itself evaluates the critic row by row, as the natural host does).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from v10 import mf_vcs as MF


# ------------------------------------------------------------------------------------------------------------ derangements
def sample_derangement(rng: np.random.Generator, n: int) -> tuple[np.ndarray, int]:
    """One uniform derangement of 0..n-1 by rejection from uniform permutations; returns (sigma int64, attempts)."""
    if isinstance(n, bool) or int(n) != n or n < 2:
        raise ValueError("a derangement needs n >= 2")
    identity = np.arange(n)
    attempts = 0
    while True:
        attempts += 1
        sigma = rng.permutation(n)
        if not np.any(sigma == identity):
            return sigma.astype(np.int64), attempts


class DerangementStream:
    """A named numpy stream of uniform derangements with an attempt counter (cost and rejection-rate record)."""

    def __init__(self, rng: np.random.Generator, n: int):
        self.rng, self.n = rng, int(n)
        self.draws = 0
        self.attempts = 0

    def draw(self, r: int = 1) -> np.ndarray:
        out = np.empty((int(r), self.n), dtype=np.int64)
        for k in range(int(r)):
            out[k], a = sample_derangement(self.rng, self.n)
            self.attempts += a
            self.draws += 1
        return out

    def record(self) -> dict:
        return {"draws": self.draws, "attempts": self.attempts,
                "mean_attempts": (self.attempts / self.draws) if self.draws else None, "expected_mean_attempts": expected_attempts(self.n)}


def derangement_count(n: int) -> int:
    d = [1, 0]
    for k in range(2, n + 1):
        d.append((k - 1) * (d[-1] + d[-2]))
    return d[n]


def expected_attempts(n: int) -> float:
    """n! / D_n (geometric mean number of uniform permutations per accepted derangement; -> e)."""
    return float(math.factorial(n) / derangement_count(n)) if n <= 170 else math.e


def all_permutations(n: int) -> np.ndarray:
    return np.array(list(itertools.permutations(range(n))), dtype=np.int64)


def all_derangements(n: int) -> np.ndarray:
    p = all_permutations(n)
    return p[~np.any(p == np.arange(n), axis=1)]


def is_derangement_matrix(sigmas: np.ndarray, n: int) -> bool:
    sigmas = np.asarray(sigmas)
    return bool(sigmas.ndim == 2 and sigmas.shape[1] == n and np.array_equal(np.sort(sigmas, axis=1), np.broadcast_to(np.arange(n), sigmas.shape))
                and not np.any(sigmas == np.arange(n)))


# ------------------------------------------------------------------------------------------------------------ counts / tables
@dataclass(frozen=True)
class Counts:
    """Sufficient statistics of n observations on an m x m support (int64)."""
    n: int
    m: int
    nxy: np.ndarray

    @classmethod
    def from_xy(cls, x: np.ndarray, y: np.ndarray, m: int) -> "Counts":
        x, y = np.asarray(x, dtype=np.int64), np.asarray(y, dtype=np.int64)
        if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
            raise ValueError("need paired 1-d observations, n >= 2")
        return cls(int(len(x)), int(m), np.bincount(x * m + y, minlength=m * m).reshape(m, m).astype(np.int64))

    @property
    def nx(self) -> np.ndarray:
        return self.nxy.sum(1)

    @property
    def ny(self) -> np.ndarray:
        return self.nxy.sum(0)

    def p_hat(self) -> np.ndarray:
        return self.nxy / self.n

    def q_hat(self, mode: str) -> np.ndarray:
        outer = np.outer(self.nx, self.ny).astype(np.float64)
        if mode == "u":
            return (outer - self.nxy) / (self.n * (self.n - 1))
        if mode == "v":
            return outer / (self.n * self.n)
        raise ValueError("mode must be 'u' or 'v'")

    def sha256(self) -> str:
        import hashlib
        return hashlib.sha256(np.ascontiguousarray(self.nxy, dtype="<i8").tobytes()).hexdigest()


def table_terms(T: np.ndarray, wp: np.ndarray, wq: np.ndarray) -> dict:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != wp.shape or T.shape != wq.shape or not np.all(np.isfinite(T)):
        raise ValueError("critic table must be finite with the weight shape")
    p1, p2 = float(np.sum(wp * T)), float(np.sum(wp * T * T))
    q1, q2 = float(np.sum(wq * T)), float(np.sum(wq * T * T))
    return {"p1": p1, "p2": p2, "q1": q1, "q2": q2, "j": p1 - q1 - 0.5 * (p2 + q2)}


def empirical_j(T: np.ndarray, counts: Counts, mode: str = "u") -> float:
    """The complete empirical J of a frozen critic table (U: all n(n-1) off-diagonal pairs; V: all n^2 pairs).  Negative values kept."""
    return table_terms(T, counts.p_hat(), counts.q_hat(mode))["j"]


def saturated_table(counts: Counts, mode: str = "u") -> np.ndarray:
    """Decision 9.5: T*_xy = (p_hat - q_hat) / (p_hat + q_hat), 0 where the denominator is 0 - the maximiser of the empirical J over all
    m x m tables (pointwise concave quadratic; p_hat, q_hat >= 0 so T* is in [-1, 1])."""
    p, q = counts.p_hat(), counts.q_hat(mode)
    den = p + q
    return np.divide(p - q, den, out=np.zeros_like(den), where=den > 0)


def perm_estimate_from_table(T: np.ndarray, x: np.ndarray, y: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    """perm1 J estimate of a frozen critic table for each derangement row of sigmas [R, n] -> [R] (float64)."""
    T = np.asarray(T, dtype=np.float64)
    tp = T[x, y]
    pos = float(np.mean(tp - 0.5 * tp * tp))
    tq = T[x[None, :], y[sigmas]]
    return pos - np.mean(tq + 0.5 * tq * tq, axis=1)


# ------------------------------------------------------------------------------------------------------------ torch objectives
def terms_from(p1: Tensor, p2: Tensor, q1: Tensor, q2: Tensor) -> MF.ObjectiveTerms:
    j = p1 - q1 - 0.5 * (p2 + q2)
    zero = j.new_zeros(())
    return MF.ObjectiveTerms(p1, p2, q1, q2, j, zero, j)


def mf_u_scene(A: Tensor, B: Tensor) -> MF.SceneBatch:
    return MF.SceneBatch(A, B, q_mode="u")


def mf_u_moments(A: Tensor, B: Tensor, W: Tensor, *, check_w: bool = True) -> MF.ObjectiveTerms:
    """MF_VCS_M0: complete U moments through the unchanged package backend (one scene, weight 1, ridge 0)."""
    return MF.objective_terms([mf_u_scene(A, B)], W, weights=[1.0], ridge=0.0, check_w=check_w)


def mf_derangement_terms(A: Tensor, B: Tensor, W: Tensor, sigmas: Tensor) -> MF.ObjectiveTerms:
    """MF_VCS_perm1 (sigmas [1, n]) / an R-derangement MF comparator: P on (a_i, b_i), Q on (a_i, b_sigma_r(i)), all R n negatives
    averaged."""
    if sigmas.ndim != 2 or sigmas.shape[1] != len(A) or sigmas.dtype != torch.long:
        raise ValueError("sigmas must be int64 [R, n]")
    AW = A @ W
    t = (AW * B).sum(-1)
    neg = (AW[None, :, :] * B[sigmas]).sum(-1)
    return terms_from(t.mean(), t.square().mean(), neg.mean(), neg.square().mean())


def mlp_derangement_terms(critic, xo: Tensor, yo: Tensor, sigmas: Tensor) -> MF.ObjectiveTerms:
    """MLP_VCS_perm1 / perm8: T = tanh f(concat(onehot x, onehot y)); P on the observed pairs, Q on R derangements (row 0 = the
    perm1 draw of the step for perm8), all R n negatives averaged - the host's VCS loss layout (P.loss_fn 'vcs' = -j_value)."""
    if sigmas.ndim != 2 or sigmas.shape[1] != len(xo) or sigmas.dtype != torch.long:
        raise ValueError("sigmas must be int64 [R, n]")
    R, n = sigmas.shape
    tp = torch.tanh(critic(xo, yo))
    xq = xo[None].expand(R, n, xo.shape[1]).reshape(R * n, -1)
    tq = torch.tanh(critic(xq, yo[sigmas.reshape(-1)]))
    return terms_from(tp.mean(), tp.square().mean(), tq.mean(), tq.square().mean())


def dense_u_terms_rows(T_pairs: Tensor) -> MF.ObjectiveTerms:
    """Reference only (tests / gates): complete U from an explicit n x n score matrix T_pairs[i, j] = T(x_i, y_j)."""
    n = T_pairs.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=T_pairs.device)
    tp = torch.diagonal(T_pairs)
    tq = T_pairs[off]
    return terms_from(tp.mean(), tp.square().mean(), tq.mean(), tq.square().mean())


def dense_v_terms_rows(T_pairs: Tensor) -> MF.ObjectiveTerms:
    tp = torch.diagonal(T_pairs)
    return terms_from(tp.mean(), tp.square().mean(), T_pairs.mean(), T_pairs.square().mean())
