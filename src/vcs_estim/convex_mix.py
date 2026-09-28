"""Estimator package v1 (§7): fixed-output combinations fitted on TUNE and applied unchanged to EVAL.

- VCS dictionary: max_w d'w - w'Gw on the simplex (exact active-face search of the package's reference core, m <= 8).
- Residual step: lambda = clip(A / B, 0, 1) on TUNE (reference core); lambda = 0 with a recorded reason when B is below tolerance.
- JS dictionary control: w on the simplex minimising the native balanced log loss of q_w = sum_j w_j sigmoid(2 f_j); log q_w and
  log(1 - q_w) by weighted log-sum-exp (no probability clipping).  Solved by SLSQP (convex problem) with a KKT check — it is *not* the
  exact QP and is named accordingly.
Inference uses T_w = sum_j w_j T_j directly (no atanh -> tanh round trip, no extra clipping).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from .bounded_core import ResidualStep, SimplexFit, fit_small_simplex, residual_step  # noqa: F401

__all__ = ["fit_small_simplex", "residual_step", "fit_js_mixture", "js_mixture_q", "SimplexFit", "ResidualStep"]


def _log_sig(z):
    return -np.logaddexp(0.0, -z)


def js_mixture_logq(F_: np.ndarray, w: np.ndarray):
    """F_: [n, m] un-squashed outputs.  Returns (log q_w, log(1 - q_w))."""
    lp, ln = _log_sig(2 * F_), _log_sig(-2 * F_)
    return logsumexp(lp, axis=1, b=w[None, :]), logsumexp(ln, axis=1, b=w[None, :])


def js_mixture_q(F_: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.exp(js_mixture_logq(F_, w)[0])


def js_mixture_loss(Fp, Fn, w) -> float:
    return float(-js_mixture_logq(Fp, w)[0].mean() - js_mixture_logq(Fn, w)[1].mean())


def fit_js_mixture(Fp: np.ndarray, Fn: np.ndarray, tol: float = 1e-10) -> dict:
    Fp, Fn = np.asarray(Fp, np.float64), np.asarray(Fn, np.float64); m = Fp.shape[1]
    sp, sn = expit(2 * Fp), expit(-2 * Fn)

    def fg(w):
        w = np.maximum(w, 0.0)
        qp, qn = sp @ w, sn @ w           # q_w on P pairs, 1 - q_w on Q pairs (for the gradient only)
        lqp, l1qn = js_mixture_logq(Fp, w)[0], js_mixture_logq(Fn, w)[1]
        val = -lqp.mean() - l1qn.mean()
        grad = -(sp / qp[:, None]).mean(0) - (sn / qn[:, None]).mean(0)
        return val, grad

    best = None
    for w0 in [np.full(m, 1.0 / m)] + [np.eye(m)[j] * 0.9 + 0.1 / m for j in range(m)]:
        r = minimize(lambda w: fg(w)[0], w0, jac=lambda w: fg(w)[1], method="SLSQP", bounds=[(0, 1)] * m,
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(m)}], options={"ftol": 1e-14, "maxiter": 500})
        w = np.maximum(r.x, 0); w /= w.sum(); v = fg(w)[0]
        if best is None or v < best[1]:
            best = (w, v)
    w, v = best; g = fg(w)[1]
    kkt = float(w @ g - g.min())          # convex minimisation on the simplex: 0 at the optimum
    return {"weights": w.tolist(), "objective_native_logloss": v, "kkt_gap": kkt, "solver": "SLSQP (convex, not the exact QP)"}
