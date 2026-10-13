"""V7X estimator module: closed forms, identities and guards (synthetic only)."""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "v7x"))
import x_estimators as XE  # noqa: E402


@pytest.mark.parametrize("shift", [0.0, 0.3, 1.0, 4.0])
def test_gaussian_equal_cov_cs(shift):
    d = XE.cs_mixture([1], [np.array([0.0, 0.0])], [np.eye(2)], [1], [np.array([shift, 0.0])], [np.eye(2)])
    assert d == pytest.approx(shift ** 2 / 2, abs=1e-12)


def test_single_centre_kde_equals_gaussian():
    p = XE.KDE([[0.0, 1.0]], [0.5, 2.0]); q = XE.KDE([[1.0, 2.0]], [2.0, 0.7])
    exact = XE.cs_mixture([1], [np.array([0.0, 1.0])], [np.diag([0.25, 4.0])], [1], [np.array([1.0, 2.0])], [np.diag([4.0, 0.49])])
    assert XE.cs_kde(p, q) == pytest.approx(exact, rel=1e-10) and XE.cs_kde(p, q) == pytest.approx(XE.cs_kde(q, p), rel=1e-12)


def test_identical_kde_zero_and_blocks():
    rng = np.random.default_rng(0); c = rng.normal(size=(9, 3)); w = rng.uniform(0.1, 1, 9)
    p1, p2 = XE.KDE(c, 0.7, w, block=1), XE.KDE(c, 0.7, w, block=64)
    assert XE.cs_kde(p1, p1) == pytest.approx(0, abs=1e-12)
    z = rng.normal(size=(5, 3)); assert np.allclose(p1.logpdf(z), p2.logpdf(z))


def test_far_apart_log_domain():
    p, q = XE.KDE([[0.0] * 30], 1.0), XE.KDE([[30.0] * 30], 1.0); v = XE.cs_kde(p, q)
    assert np.isfinite(v) and v == pytest.approx(30 * 30 ** 2 / 2, rel=1e-9)


def test_mixture_cs_against_quadrature():
    ap, mp, cp = [0.3, 0.7], [np.array([-1.0]), np.array([1.5])], [np.eye(1) * 0.5, np.eye(1) * 1.2]
    aq, mq, cq = [0.5, 0.5], [np.array([0.0]), np.array([2.5])], [np.eye(1) * 0.8, np.eye(1) * 0.3]
    x = np.linspace(-12, 14, 200001); dx = x[1] - x[0]
    pdf = lambda a, m, c: sum(ai * np.exp(-0.5 * (x - mi[0]) ** 2 / ci[0, 0]) / math.sqrt(2 * math.pi * ci[0, 0]) for ai, mi, ci in zip(a, m, c))
    P, Q = pdf(ap, mp, cp), pdf(aq, mq, cq); num = -2 * math.log((P * Q).sum() * dx) + math.log((P * P).sum() * dx) + math.log((Q * Q).sum() * dx)
    assert XE.cs_mixture(ap, mp, cp, aq, mq, cq) == pytest.approx(num, rel=1e-6)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75])
def test_relative_ratio_categorical(alpha):
    p, q, e = np.array([0.7, 0.3]), np.array([0.2, 0.8]), np.eye(2)
    assert np.allclose(XE.relative_ratio_fit(e, e, alpha, 0.0, p, q), p / (alpha * p + (1 - alpha) * q))


def test_relative_inverse_and_readout():
    p, q = np.array([0.7, 0.3]), np.array([0.2, 0.8]); u = p / (0.5 * p + 0.5 * q)
    assert np.allclose(XE.relative_to_ratio(u, 0.5), p / q)
    r = XE.relative_half_readout([-1.0, 0.5, 3.0]); assert np.array_equal(r["u_raw"], [-1.0, 0.5, 3.0]) and np.array_equal(r["T"], [-1.0, -0.5, 1.0])
    assert r["clip_fraction"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        XE.relative_to_ratio(u, 0.0)


def test_squared_posterior_categorical():
    p, q, e = np.array([0.7, 0.3]), np.array([0.2, 0.8]), np.eye(2)
    assert np.allclose(XE.squared_posterior_fit(e, e, 0.0, p, q), (p - q) / (p + q))


@pytest.mark.parametrize("shift", [-80.0, 0.0, 60.0])
def test_dv_gauge_and_nwj(shift):
    p, q = np.array([0.7, 0.3]), np.array([0.2, 0.8]); f = np.log(p / q) + shift
    out = XE.dv_readout(f, XE.dv_log_normaliser(f, q)); assert np.allclose(out["d"], p / (p + q))
    n = XE.nwj_readout(1 + np.log(p / q), 0.3); assert np.allclose(n["p_nat"], 0.3 * p / (0.3 * p + 0.7 * q))


def test_nystrom_whitening_reproduces_kernel():
    rng = np.random.default_rng(1); c = rng.normal(size=(40, 3)); nm = XE.nystrom_map(c, 1.3, eps_rel=1e-10)
    phi = XE.nystrom_features(c, nm); K = np.exp(-0.5 * np.maximum(XE.sq_dist(c, c), 0) / 1.3 ** 2)
    assert np.allclose(phi @ phi.T, K, atol=1e-6)


def test_soft_masses_and_guards():
    a, b, pi = XE.soft_masses(np.array([0.25, 0.8, 0.5]), np.array([1.0, 2.0, 1.0]))
    assert pi == pytest.approx((0.25 + 1.6 + 0.5) / 4) and a.sum() == pytest.approx(1) and b.sum() == pytest.approx(1)
    for bad in ([1.0, 1.0], [0.0, 0.0], [-0.1, 0.5], [np.nan, 0.5]):
        with pytest.raises(ValueError):
            XE.soft_masses(np.array(bad))
    for h in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError):
            XE.KDE([[1.0, 2.0]], h)


def test_ess():
    assert XE.effective_sample_size(np.zeros(10)) == pytest.approx(10)
    assert XE.effective_sample_size(np.array([0.0, -1e9, -1e9])) == pytest.approx(1)


@pytest.mark.skipif(not os.environ.get("V7X_REF_DIR"), reason="package reference copy not given (V7X_REF_DIR)")
def test_cross_check_with_package_reference():
    """Numerical agreement with the owner's v2 reference implementation on random inputs (the reference is not part of this repository)."""
    sys.path.insert(0, os.environ["V7X_REF_DIR"]); import v7_external_estimators as R  # noqa: E402
    rng = np.random.default_rng(7); x, y, z = rng.normal(size=(12, 4)), rng.normal(size=(9, 4)) + 0.5, rng.normal(size=(6, 4))
    wa, wb = rng.uniform(0.1, 1, 12), rng.uniform(0.1, 1, 9)
    ours_p, ours_q, ref_p, ref_q = XE.KDE(x, 0.8, wa), XE.KDE(y, [0.6, 0.9, 1.1, 0.7], wb), R.GaussianKDE(x, 0.8, wa), R.GaussianKDE(y, [0.6, 0.9, 1.1, 0.7], wb)
    assert np.allclose(ours_p.logpdf(z), ref_p.logpdf(z)) and XE.cs_kde(ours_p, ours_q) == pytest.approx(R.native_cs_kde(ref_p, ref_q), rel=1e-10)
    phi_p, phi_q = rng.uniform(size=(20, 5)), rng.uniform(size=(15, 5)); wp, wq = rng.uniform(size=20), rng.uniform(size=15)
    assert np.allclose(XE.relative_ratio_fit(phi_p, phi_q, 0.5, 1e-3, wp, wq), R.fit_relative_ratio(phi_p, phi_q, 0.5, 1e-3, wp, wq))
    assert np.allclose(XE.squared_posterior_fit(phi_p, phi_q, 1e-3, wp, wq), R.fit_kernel_basis_posterior(phi_p, phi_q, 1e-3, wp, wq))
    f = rng.normal(size=8); assert XE.dv_log_normaliser(f) == pytest.approx(R.dv_log_normalizer(f))
    assert np.allclose(XE.nwj_readout(f, 0.4)["p_nat"], R.nwj_readout(f, 0.4)["natural_probability"])
    assert np.allclose(XE.relative_half_readout([0.3, 2.5], 0.4)["p_nat"], R.relative_half_readout([0.3, 2.5], 0.4)["natural_probability"])
