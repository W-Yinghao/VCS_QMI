"""Oracle-critic resolution of the dependence strength for the bivariate Gaussian (re-derivation of CS_QMI/RESULTS_20260928.md, single
shuffled negative, n joint pairs): δρ = SD(estimate) / |dE[estimate]/dρ|, and the Cramér–Rao bound for ρ.

At the oracle critic the estimate is a mean of n i.i.d. summands over the joint sample plus a mean of n i.i.d. summands over an independent
product sample, so Var = (Var_P[a₊] + Var_Q[a₋]) / n and E is the population value.  Diagonalising the quadratic form of the PMI gives
  under P:  pmi = c + (ρ/2)(u² − v²),                         under Q:  pmi = c + ρ/(2(1+ρ)) · u² − ρ/(2(1−ρ)) · v²,
with u, v i.i.d. N(0, 1) and c = −½ log(1 − ρ²).  Moments are Monte Carlo averages with common random numbers across ρ (the derivative is a
central difference on the same draws); the NWJ product-side moments, which are rare-event dominated (E_Q[e^{2·pmi}] = 1/(1 − ρ²)), are analytic.
"""
from __future__ import annotations

import math

import numpy as np


def _pmi_P(rho, u, v):
    return -0.5 * math.log(1 - rho * rho) + 0.5 * rho * (u * u - v * v)


def _pmi_Q(rho, u, v):
    return -0.5 * math.log(1 - rho * rho) + rho / (2 * (1 + rho)) * u * u - rho / (2 * (1 - rho)) * v * v


def _summands(kind, p, side):
    if kind == "vcs":
        T = np.tanh(p / 2); return T - T * T / 2 if side == "P" else -T - T * T / 2
    if kind == "js":
        return -np.logaddexp(0, -p) if side == "P" else -np.logaddexp(0, p)
    if kind == "nwj":
        return 1 + p if side == "P" else -np.exp(p)
    raise ValueError(kind)


def estimate_moments(kind: str, rho: float, u: np.ndarray, v: np.ndarray):
    """(E, Var of the sum of one P summand and one Q summand) on the common draws (u, v)."""
    aP = _summands(kind, _pmi_P(rho, u, v), "P"); mP, vP = aP.mean(), aP.var()
    if kind == "nwj":
        mQ, vQ = -1.0, rho * rho / (1 - rho * rho)
    else:
        aQ = _summands(kind, _pmi_Q(rho, u, v), "Q"); mQ, vQ = aQ.mean(), aQ.var()
    return mP + mQ, vP + vQ


def resolution(kind: str, rho: float, n: int = 10_000, n_mc: int = 4_000_000, seed: int = 0, h: float | None = None) -> float:
    rng = np.random.default_rng(seed); u, v = rng.standard_normal(n_mc), rng.standard_normal(n_mc)
    h = h or (1 - rho) / 200
    e_plus, _ = estimate_moments(kind, rho + h, u, v); e_minus, _ = estimate_moments(kind, rho - h, u, v); _, var = estimate_moments(kind, rho, u, v)
    return math.sqrt(var / n) / abs((e_plus - e_minus) / (2 * h))


def crb(rho: float, n: int = 10_000) -> float:
    """Cramér–Rao bound for rho of a bivariate normal with known unit variances: I(rho) = (1 + rho²)/(1 − rho²)²."""
    return (1 - rho * rho) / math.sqrt(n * (1 + rho * rho))


def oracle_table(rhos=(0.9, 0.99, 0.9999, 0.999999), n: int = 10_000, n_mc: int = 4_000_000):
    rows = []
    for r in rhos:
        d = {k: resolution(k, r, n, n_mc) for k in ("vcs", "nwj", "js")}
        rng = np.random.default_rng(0); u, v = rng.standard_normal(1_000_000), rng.standard_normal(1_000_000)
        rows.append({"rho": r, "I_nats": -0.5 * math.log(1 - r * r), "S": estimate_moments("vcs", r, u, v)[0], "NWJ/VCS": d["nwj"] / d["vcs"], "JS/VCS": d["js"] / d["vcs"], "VCS/CRB": d["vcs"] / crb(r, n)})
    return rows
