"""V7X estimators of the external comparison (v2 package docs/11-12): classical fixed-reference CS with Gaussian KDE (information potentials),
closed-form Gaussian / Gaussian-mixture truths, KDE posterior, relative density ratio (RuLSIF, alpha in [0, 1); alpha = 0 is uLSIF), whitened
Nystrom squared regression, and the DV / NWJ log-ratio bridges.  Standard formulas, implemented here; NumPy for the exact small cases (tests,
controlled anchors) and torch block versions for the natural-feature fits.  Conventions: normalised Gaussian kernels; CS in the squared-ratio
form D = -2 log C + log A + log B (scale_factor 1; a 1/2 convention is recorded, never applied silently)."""
from __future__ import annotations

import math

import numpy as np
from scipy.special import expit, logsumexp

SCALE_FACTOR = 1.0


# ------------------------------------------------------------------------------------------------------------------------------- checks
def weights(w, n: int) -> np.ndarray:
    a = np.ones(n) if w is None else np.asarray(w, dtype=np.float64)
    if a.shape != (n,) or not np.all(np.isfinite(a)) or np.any(a < 0) or a.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, of positive total and of length n")
    return a / a.sum()


def matrix(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0 or not np.all(np.isfinite(a)):
        raise ValueError("a non-empty finite 2-d array is required")
    return a


def bandwidth(h, d: int) -> np.ndarray:
    a = np.broadcast_to(np.asarray(h, dtype=np.float64), (d,)).copy()
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError("bandwidths must be positive and finite")
    return a


def logw(w):
    with np.errstate(divide="ignore"):
        return np.log(w)


# ------------------------------------------------------------------------------------------------------------------------------ kernels
def log_gauss(x, y, h) -> np.ndarray:
    """log N(x_i - y_j; 0, diag(h^2)) for all pairs, shape [n_x, n_y]."""
    x, y = matrix(x), matrix(y)
    if x.shape[1] != y.shape[1]:
        raise ValueError("dimension mismatch")
    h = bandwidth(h, x.shape[1]); xs, ys = x / h, y / h
    d2 = (xs ** 2).sum(1)[:, None] + (ys ** 2).sum(1)[None, :] - 2 * xs @ ys.T
    return -0.5 * np.maximum(d2, 0) - np.log(h).sum() - 0.5 * x.shape[1] * math.log(2 * math.pi)


class KDE:
    """Weighted Gaussian KDE with a diagonal bandwidth (exact, NumPy, blocks of rows)."""

    def __init__(self, centres, h, w=None, block=256):
        self.c = matrix(centres); self.h = bandwidth(h, self.c.shape[1]); self.w = weights(w, len(self.c)); self.block = int(block)
        if self.block <= 0:
            raise ValueError("block must be positive")

    def logpdf(self, x) -> np.ndarray:
        x = matrix(x); lw = logw(self.w)
        return np.concatenate([logsumexp(log_gauss(x[i:i + self.block], self.c, self.h) + lw[None, :], axis=1) for i in range(0, len(x), self.block)])

    def log_overlap(self, other: "KDE") -> float:
        """log of the integral of p_hat q_hat (Gaussian convolution: bandwidth sqrt(h_p^2 + h_q^2)); diagonal pairs included."""
        if self.c.shape[1] != other.c.shape[1]:
            raise ValueError("dimension mismatch")
        h = np.sqrt(self.h ** 2 + other.h ** 2); l1, l2 = logw(self.w), logw(other.w)
        parts = [logsumexp(log_gauss(self.c[i:i + self.block], other.c, h) + l1[i:i + self.block, None] + l2[None, :]) for i in range(0, len(self.c), self.block)]
        return float(logsumexp(parts))


def cs_kde(p: KDE, q: KDE) -> float:
    """Classical (Lebesgue) CS divergence of the fitted KDE densities, squared-ratio form, diagonal included; not clipped."""
    return float(SCALE_FACTOR * (-2 * p.log_overlap(q) + p.log_overlap(p) + q.log_overlap(q)))


def gauss_log_overlap(mp, cp, mq, cq) -> float:
    """log of the integral of N(mp, cp) N(mq, cq) = log N(mp; mq, cp + cq)."""
    mp, mq, cp, cq = (np.asarray(v, np.float64) for v in (mp, mq, cp, cq))
    if mp.ndim != 1 or mp.shape != mq.shape or cp.shape != (len(mp), len(mp)) or cq.shape != cp.shape:
        raise ValueError("shape mismatch")
    for c in (cp, cq):
        if not np.allclose(c, c.T):
            raise ValueError("covariance must be symmetric")
    L = np.linalg.cholesky(cp + cq); r = np.linalg.solve(L, mp - mq)
    return float(-0.5 * (len(mp) * math.log(2 * math.pi) + 2 * np.log(np.diag(L)).sum() + r @ r))


def mixture_log_overlap(ap, mps, cps, aq, mqs, cqs) -> float:
    """log of the integral of p q for Gaussian mixtures (component weights ap / aq)."""
    ap, aq = weights(ap, len(mps)), weights(aq, len(mqs))
    t = [math.log(ap[i]) + math.log(aq[j]) + gauss_log_overlap(mps[i], cps[i], mqs[j], cqs[j]) for i in range(len(mps)) for j in range(len(mqs))]
    return float(logsumexp(t))


def cs_mixture(ap, mps, cps, aq, mqs, cqs) -> float:
    """Closed-form classical CS between two Gaussian mixtures (single Gaussians: one component each)."""
    return float(SCALE_FACTOR * (-2 * mixture_log_overlap(ap, mps, cps, aq, mqs, cqs) + mixture_log_overlap(ap, mps, cps, ap, mps, cps)
                                 + mixture_log_overlap(aq, mqs, cqs, aq, mqs, cqs)))


# ------------------------------------------------------------------------------------------------------------------------------ readouts
def ratio_readout(ell, prior: float = 0.5) -> dict:
    """Balanced log ratio ell = log p / q -> T = tanh(ell / 2), d = sigmoid(ell), natural p = sigmoid(ell + logit prior)."""
    ell = np.asarray(ell, dtype=np.float64)
    if not np.all(np.isfinite(ell)) or not 0 < prior < 1:
        raise ValueError("finite log ratios and a prior in (0, 1) are required")
    return {"ell": ell, "T": np.tanh(ell / 2), "d": expit(ell), "p_nat": expit(ell + math.log(prior / (1 - prior)))}


def soft_masses(y, base=None):
    """Soft votes -> normalised P mass (w y), Q mass (w (1 - y)) and the prior (w-weighted mean y); both classes must have mass."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or len(y) == 0 or not np.all(np.isfinite(y)) or np.any((y < 0) | (y > 1)):
        raise ValueError("soft votes must be a non-empty vector in [0, 1]")
    w = weights(base, len(y)); prior = float(w @ y)
    if not 0 < prior < 1:
        raise ValueError("both classes need mass")
    return weights(w * y, len(y)), weights(w * (1 - y), len(y)), prior


def relative_ratio_fit(phi_p, phi_q, alpha=0.5, ridge=1e-4, wp=None, wq=None):
    """theta = (H + ridge I)^-1 h with H = alpha E_P[phi phi'] + (1 - alpha) E_Q[phi phi'], h = E_P[phi] (unconstrained least squares;
    alpha = 0 is uLSIF).  The author's CV and non-negativity options are separate, recorded choices."""
    p, q = matrix(phi_p), matrix(phi_q)
    if p.shape[1] != q.shape[1] or not 0 <= alpha < 1 or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("invalid relative-ratio parameters")
    wp, wq = weights(wp, len(p)), weights(wq, len(q))
    H = alpha * (p.T * wp) @ p + (1 - alpha) * (q.T * wq) @ q
    return np.linalg.solve(H + ridge * np.eye(p.shape[1]), p.T @ wp)


def relative_half_readout(u, prior=0.5) -> dict:
    """alpha = 1/2: u = 2p / (p + q) in [0, 2]; raw u kept, the projection onto [0, 2] gives d = u / 2 (not guaranteed by finite fits)."""
    raw = np.asarray(u, dtype=np.float64)
    if not np.all(np.isfinite(raw)) or not 0 < prior < 1:
        raise ValueError("finite u and a prior in (0, 1) are required")
    cl = np.clip(raw, 0, 2); d = cl / 2; pn = prior * d / (prior * d + (1 - prior) * (1 - d))
    return {"u_raw": raw, "u": cl, "T": cl - 1, "d": d, "p_nat": pn, "clip_fraction": float(np.mean(raw != cl))}


def relative_to_ratio(u, alpha):
    """p / q = (1 - alpha) u / (1 - alpha u) for 0 <= u < 1 / alpha (alpha in (0, 1)); u = 1 / alpha is an infinite ratio (returned as inf)."""
    u = np.asarray(u, dtype=np.float64)
    if not 0 < alpha < 1:
        raise ValueError("the inverse map needs 0 < alpha < 1 (alpha = 0: u is already p / q)")
    with np.errstate(divide="ignore"):
        return np.where(u >= 1 / alpha, np.inf, (1 - alpha) * u / (1 - alpha * u))


def squared_posterior_fit(phi_p, phi_q, ridge=1e-4, wp=None, wq=None):
    """min 1/2 E_P (1 - T)^2 + 1/2 E_Q (-1 - T)^2 + ridge |theta|^2 over T = phi theta; with whitened Nystrom features the ridge is the
    approximate RKHS norm."""
    p, q = matrix(phi_p), matrix(phi_q)
    if p.shape[1] != q.shape[1] or not np.isfinite(ridge) or ridge < 0:
        raise ValueError("invalid parameters")
    wp, wq = weights(wp, len(p)), weights(wq, len(q))
    H = 0.5 * ((p.T * wp) @ p + (q.T * wq) @ q)
    return np.linalg.solve(H + ridge * np.eye(p.shape[1]), 0.5 * (p.T @ wp - q.T @ wq))


def nystrom_map(centres, h, eps_rel=1e-6):
    """Whitening for Phi(x) = K(x, C) (K_CC + eps I)^(-1/2) with an unnormalised Gaussian kernel exp(-|x - c|^2 / (2 h^2)); returns (C, h, W)."""
    c = matrix(centres); h = float(h); K = np.exp(-0.5 * np.maximum(sq_dist(c, c), 0) / h ** 2); eps = eps_rel * float(np.mean(np.diag(K)))
    vals, vecs = np.linalg.eigh(K + eps * np.eye(len(c))); W = (vecs / np.sqrt(vals)) @ vecs.T
    return c, h, W


def nystrom_features(x, nmap):
    c, h, W = nmap
    return np.exp(-0.5 * np.maximum(sq_dist(matrix(x), c), 0) / h ** 2) @ W


def sq_dist(x, y):
    return (x ** 2).sum(1)[:, None] + (y ** 2).sum(1)[None, :] - 2 * x @ y.T


def dv_log_normaliser(f_q, wq=None) -> float:
    """log E_Q[e^f] on an independent reference sample (once per fitted f, then frozen)."""
    f = np.asarray(f_q, dtype=np.float64)
    if f.ndim != 1 or len(f) == 0 or not np.all(np.isfinite(f)):
        raise ValueError("finite reference values required")
    return float(logsumexp(f + logw(weights(wq, len(f)))))


def dv_readout(f, log_norm, prior=0.5):
    if not np.isfinite(log_norm):
        raise ValueError("non-finite normaliser")
    return ratio_readout(np.asarray(f, np.float64) - log_norm, prior)


def nwj_readout(f, prior=0.5):
    """NWJ optimum f* = 1 + log p / q -> ell = f - 1 (raw bridge; no offset re-estimation)."""
    return ratio_readout(np.asarray(f, np.float64) - 1.0, prior)


def effective_sample_size(logw_):
    """Kish ESS of normalised exp-weights (DV tail diagnostic)."""
    lw = np.asarray(logw_, np.float64); lw = lw - logsumexp(lw); return float(1.0 / np.exp(logsumexp(2 * lw)))
