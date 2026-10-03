"""P118 shared pieces (v5 NEXT-I-NEST): a Gaussian conditional-binary generator with an exact oracle, nested chain fitting and the evaluation
of increments / posterior errors with sampled or exactly enumerated product terms.  Uses src/vcs_measure/nested.py unchanged.

Generator (named numpy streams).  Y uniform on 10 classes; P(N = 1 | Y) = ½ ± 0.3 by class parity (the T1 / I1 design).  z ∈ R^64:
    z = mu_Y + (2N − 1) · (delta / 2) · v + eps,   eps ~ N(0, I),  mu_y ~ N(0, I) per class (fixed),  v = 1/sqrt(8) on coordinates 0..7, 0 elsewhere.
Coordinates 8..63 are independent of N given Y and of the relevant block.  Information chain (coordinate deletion, deterministic):
    B = z (64)  ⊇  M = z[:, :8]  (drops only irrelevant coordinates: S_B = S_M exactly)  ⊇  A = z[:, :4]  (drops half of the relevant block: S_A < S_M).
Oracle for a kept block K with a_K = (delta / 2) v_K and t = a_K · (z_K − mu_{Y,K}):  log N(z | n = 1) − log N(z | n = 0) = 2t, so
    r(z, n, y) = p(z | n, y) / p(z | y) = 1 / (p1 + p0 e^{−2t}) for n = 1,  1 / (p1 e^{2t} + p0) for n = 0;   eta = (r − 1) / (r + 1).
P units (z, n, y); Q units (z, n', y) with n' ~ P(N | y) independent of z given y.  S_K = E_M eta_K², estimated on a large truth sample.
"""
from __future__ import annotations

import numpy as np
import torch

D, R, KEEP_A, N_CLASSES, P_SHIFT = 64, 8, 4, 10, 0.3
SETS = {"B": D, "M": R, "A": KEEP_A}
CHAIN = ["B", "M", "A"]


def p_n1(y):
    return 0.5 + P_SHIFT * np.where(np.asarray(y) % 2 == 0, 1.0, -1.0)


def class_means(seed: int = 118) -> np.ndarray:
    return np.random.default_rng([seed, 1]).normal(0.0, 1.0, (N_CLASSES, D))


def sample(n: int, delta: float, rng, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = rng.integers(0, N_CLASSES, n); nn_ = (rng.random(n) < p_n1(y)).astype(np.int64)
    v = np.zeros(D); v[:R] = 1.0 / np.sqrt(R)
    z = mu[y] + (2 * nn_ - 1)[:, None] * (0.5 * delta) * v[None, :] + rng.normal(0.0, 1.0, (n, D))
    return z, y, nn_


def eta_set(z: np.ndarray, n: np.ndarray, y: np.ndarray, delta: float, mu: np.ndarray, k: int) -> np.ndarray:
    """Exact eta of the kept block z[:, :k] (k ∈ {64, 8, 4}); coordinates >= 8 never enter (they cancel)."""
    kk = min(k, R)
    if delta == 0 or kk == 0:
        return np.zeros(len(z))
    a = (0.5 * delta / np.sqrt(R)) * np.ones(kk); t = (z[:, :kk] - mu[y][:, :kk]) @ a
    p1 = p_n1(y); p0 = 1 - p1
    r = np.where(n == 1, 1.0 / (p1 + p0 * np.exp(-2 * t)), 1.0 / (p1 * np.exp(2 * t) + p0))
    return (r - 1) / (r + 1)


def oracle_S(delta: float, n: int = 400_000, seed: int = 1180) -> dict:
    """S_K for K ∈ {B, M, A} on one truth sample (P units and an independent-n' Q copy of the same z), with sampling SEs."""
    mu = class_means(); rng = np.random.default_rng([seed, 2]); z, y, nn_ = sample(n, delta, rng, mu)
    nq = (rng.random(n) < p_n1(y)).astype(np.int64); out = {}
    for name, k in SETS.items():
        ep, eq = eta_set(z, nn_, y, delta, mu, k), eta_set(z, nq, y, delta, mu, k); sp, sq = ep ** 2, eq ** 2
        out[name] = {"S": float(0.5 * sp.mean() + 0.5 * sq.mean()), "S_se": float(np.sqrt(0.25 * sp.var() / n + 0.25 * sq.var() / n))}
    return out


def zero_readouts(n: int) -> tuple:
    """Constant-zero critic: T ≡ 0 on P, sampled Q [n, 1] and enumerated Q [n, 2]."""
    return np.zeros(n), (np.zeros((n, 1)), np.ones((n, 1))), (np.zeros((n, 2)), None)


def posterior_mse(tp: np.ndarray, tq: np.ndarray, wq: np.ndarray, ep: np.ndarray, eq: np.ndarray) -> float:
    """½ E_P (T − eta)² + ½ E_Q (T − eta)², the Q expectation with the readout's weights (sampled partner or exact enumeration)."""
    return float(0.5 * np.mean((tp - ep) ** 2) + 0.5 * np.mean(np.sum(wq * (tq - eq) ** 2, axis=1)))


def to_t(a, device):
    return torch.as_tensor(a).to(device)
