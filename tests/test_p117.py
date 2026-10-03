"""P117 (v5 NEXT-E-CAL) tests: the calibration decomposition on an exact discrete toy (float64), bounded calibrators, CV selection uses CAL only,
CAL / EVAL disjointness."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim import p117  # noqa: E402
from vcs_estim.benchmark import Role, build_roles  # noqa: E402
from vcs_estim.data import setting_from_mi  # noqa: E402


def toy(k=40, seed=0):
    r = np.random.default_rng(seed); p = r.dirichlet(np.ones(k)); q = r.dirichlet(np.ones(k))
    m = 0.5 * (p + q); eta = (p - q) / (p + q); return p, q, m, eta


def pop_J(g, p, q):
    return float((p * (g - 0.5 * g ** 2)).sum() + (q * (-g - 0.5 * g ** 2)).sum())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_decomposition_exact_discrete(seed):
    p, q, m, eta = toy(seed=seed); S = float((m * eta ** 2).sum())
    U = np.round(np.clip(eta + np.random.default_rng(seed + 9).normal(0, 0.3, len(eta)), -0.99, 0.99), 1)   # a lossy score with ties
    groups = np.unique(U); mU = np.empty_like(eta)
    for gv in groups:                                                       # exact E_M[eta | U]
        s = U == gv; mU[s] = (m[s] * eta[s]).sum() / m[s].sum()
    A = float((m * (eta - mU) ** 2).sum())
    la = p117.LatentAffine(); la.alpha, la.beta = 1.7, -0.2
    for g in (U.copy(), la(U), np.tanh(3 * U) * 0.9, np.zeros_like(U), mU):
        gap = S - pop_J(g, p, q); B = float((m * (g - mU) ** 2).sum())
        assert abs(gap - float((m * (eta - g) ** 2).sum())) < 1e-12          # S − J(g) = E_M (eta − g)²
        assert abs(gap - (A + B)) < 1e-12                                    # = A + B (orthogonality)
    assert abs(S - pop_J(mU, p, q) - A) < 1e-12                             # the best calibrator attains A


def test_calibrators_bounded():
    r = np.random.default_rng(0); up = np.tanh(r.normal(0.5, 2, 2000)); uq = np.tanh(r.normal(-0.5, 2, 2000)); u = np.linspace(-1, 1, 1001)
    la = p117.LatentAffine().fit(up, uq); la.alpha = 50.0
    for g in (p117.Identity(), la, p117.Bins(8).fit(up, uq), p117.Bins(32).fit(up, uq)):
        v = g(u); assert np.all(v <= 1.0) and np.all(v >= -1.0) and np.all(np.isfinite(v))


def test_bins_value_and_affine_identity():
    up = np.array([-0.9, 0.1, 0.2, 0.8]); uq = np.array([-0.8, -0.7, 0.15, 0.9])
    b = p117.Bins(2).fit(up, uq)                                             # median of pooled split
    lo_p, lo_q = (up <= b.edges[0]).mean(), (uq <= b.edges[0]).mean()
    assert np.isclose(b(np.array([-0.95]))[0], (lo_p - lo_q) / (lo_p + lo_q))
    la = p117.LatentAffine(); u = np.linspace(-0.99, 0.99, 11); assert np.allclose(la(u), u, atol=1e-9)


def test_sq_risk_is_one_minus_J():
    r = np.random.default_rng(1); gp, gq = np.tanh(r.normal(size=500)), np.tanh(r.normal(size=700))
    assert abs(p117.sq_risk(gp, gq) - (1 - p117.j_np(gp, gq))) < 1e-12


def test_cv_select_uses_cal_only():
    assert list(inspect.signature(p117.cv_select).parameters) == ["up", "uq", "seed"]
    r = np.random.default_rng(2); up, uq = np.tanh(r.normal(1, 1, 800)), np.tanh(r.normal(-1, 1, 800))
    s = p117.cv_select(up, uq, 0); assert s["selected"] in p117.CANDS and s["k_star"] in p117.BIN_COUNTS


def test_cal_eval_disjoint_and_detects_reuse():
    st = setting_from_mi("gaussian", 4.0, 20); roles, _ = build_roles(st, 20, 20, 256, 0, smoke=True)
    p117.assert_disjoint(roles["TUNE"], roles["EVAL"], "CAL/EVAL")
    D = p117.extra_role(st, 20, 20, 500, 256, 0, "P117-DIAG"); p117.assert_disjoint(D, roles["EVAL"], "DIAG/EVAL")
    with pytest.raises(AssertionError):
        p117.assert_disjoint(roles["EVAL"], roles["EVAL"], "self")
