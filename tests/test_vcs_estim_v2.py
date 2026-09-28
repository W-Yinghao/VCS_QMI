"""Estimator package v2 (server spec §12): mathematical / scale regression checks for the direct CS controls, the irrelevant-dimension
design, the combination bound, the residual pseudo-target, the noise scale, nested increments and the statistical units — plus the identities
of the package's own `support/checks_v2.py`.  Exact algebra in float64 at floating-point tolerance; Monte-Carlo checks against a predicted
standard error.  No neural training beyond a few gradient steps."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim import benchmark as BM  # noqa: E402
from vcs_estim import kernel_cs as KC  # noqa: E402
from vcs_estim import p85_grid as PG  # noqa: E402
from vcs_estim.bounded_core import fit_small_simplex  # noqa: E402
from vcs_estim.critics import JointMLP, OracleCritic  # noqa: E402
from vcs_estim.data import make_setting, setting_from_mi, truths_by_mc  # noqa: E402
from vcs_estim.estimators import KINDS, estimate, oracle_transform, scores  # noqa: E402
from vcs_estim.objectives import j_hat, js_match_loss, risk_hat, vcs_loss  # noqa: E402
from vcs_estim.oracle import estimate_moments  # noqa: E402
from vcs_estim.pairing import check_disjoint, eval_blocks, role_split  # noqa: E402
from vcs_estim.synthetic import eta, eta_signal, make_gaussian, make_gaussian_padded, rho_from_I, truth  # noqa: E402

TOL = 1e-12
RNG = np.random.default_rng(20260928)


@pytest.fixture(autouse=True, scope="module")
def _float64_default():
    """Exact algebra in float64 for this module only (the other suites keep torch's float32 default)."""
    old = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _discrete(nx=6, ny=5, seed=1):
    rng = np.random.default_rng(seed); p = rng.random((nx, ny)); p /= p.sum(); q = np.outer(p.sum(1), p.sum(0))
    return p, q, (p - q) / (p + q)


def _J(p, q, t):
    return float((p * (t - 0.5 * t * t)).sum() + (q * (-t - 0.5 * t * t)).sum())


# ------------------------------------------------------------------------------------------------- posterior & risk; initial gradient scale
def test_posterior_risk_and_initial_gradient_scale():
    p, q, e = _discrete(); M = 0.5 * (p + q)
    for _ in range(5):
        T = np.tanh(RNG.normal(size=p.shape))
        R = 0.5 * (p * (1 - T) ** 2).sum() + 0.5 * (q * (-1 - T) ** 2).sum()
        assert abs(_J(p, q, T) - (1 - R)) < TOL
        assert abs(((M * e * e).sum() - _J(p, q, T)) - (M * (T - e) ** 2).sum()) < TOL
    fp = torch.zeros(5, requires_grad=True); fn = torch.zeros(11, requires_grad=True)
    gv = torch.autograd.grad(vcs_loss(fp, fn), (fp, fn)); gj = torch.autograd.grad(js_match_loss(fp, fn), (fp, fn))
    assert all(torch.allclose(a, b, atol=1e-15) for a, b in zip(gv, gj))
    # local gate relation d ell_V / d f = (T - C)(1 - T^2) vs d ell_JS / d f = T - C  (checks_v2 finite differences)
    eps = 1e-5
    for c in (-1.0, 1.0):
        for f in (-2.0, -0.3, 0.0, 0.6, 2.0):
            lv = lambda x: 0.5 * (c - np.tanh(x)) ** 2; lj = lambda x: np.logaddexp(0.0, -2 * c * x); tv = np.tanh(f)
            assert abs((lv(f + eps) - lv(f - eps)) / (2 * eps) - (tv - c) * (1 - tv * tv)) < 1e-8
            assert abs((lj(f + eps) - lj(f - eps)) / (2 * eps) - (tv - c)) < 1e-8


# ------------------------------------------------------------------------------------------------- same-target kernel: two fields, independent roles
def test_s_kde_two_fields_on_independent_queries():
    rho, D = make_gaussian(2.0, d=3, seed=0, sizes={"FIT": 800, "TUNE": 300, "SELECT": 300, "EVAL": 2000, "TRUTH": 20000})
    F, E = D["FIT"], D["EVAL"]
    sel = KC.select_kde_bandwidth(F.xp, F.yp, D["TUNE"].xp, D["TUNE"].yp, D["TUNE"].xq, D["TUNE"].yq, multipliers=(0.5, 1.0))
    tp = KC.s_kde_eta(E.xp, E.yp, F.xp, F.yp, sel["h"], sel["b"]); tn = KC.s_kde_eta(E.xq, E.yq, F.xp, F.yp, sel["h"], sel["b"])
    assert tp.abs().max() < 1 and tn.abs().max() < 1
    r = KC.same_target_readouts(tp, tn, eta(E.xp, E.yp, rho), eta(E.xq, E.yq, rho), S=truth(2.0, 3, rho, D["TRUTH"])["S_truth"])
    # S_plug and J_kernel are distinct fields: S_plug - J_kernel = E_M (T - ... ) generally != 0; both finite; J_kernel <= S_plug is NOT implied,
    # but S_plug - J_kernel = E_M[T^2] - (2 E_M[T eta_emp] - E_M T^2) has the exact form checked below on the samples themselves
    assert abs((r["S_plug"] - r["J_kernel"]) - float(2 * (0.5 * (tp * tp).mean() + 0.5 * (tn * tn).mean()) - (tp.mean() - tn.mean()))) < 1e-12
    assert np.isfinite(r["posterior_mse"]) and r["posterior_mse"] >= 0
    # queries are independent of the support (different role streams)
    assert not torch.equal(F.xp[:5], E.xp[:5]) and not torch.equal(F.xp[:5], D["TUNE"].xp[:5])
    # log-density normalisation: integral check on a 1-d support with a wide grid
    s = torch.randn(50, 1); g = torch.linspace(-12, 12, 20001)[:, None]
    lp = KC.kde_log_density(g, s, 0.7); assert abs(float(torch.exp(lp).sum() * (g[1, 0] - g[0, 0])) - 1.0) < 1e-6


# ------------------------------------------------------------------------------------------------- native kernel CS: extremes, autodiff, analytic
def test_kernel_cs_native_extremes_autodiff_and_gaussian_truth():
    n = 17; ones = torch.ones(n, n); eye = torch.eye(n)
    assert abs(float(KC.kernel_cs_from_grams(ones, ones)["D_CS"])) < TOL
    assert abs(float(KC.kernel_cs_from_grams(eye, eye)["D_CS"]) - math.log(n)) < 1e-12         # plug-in diagonal effect, not dependence
    x = torch.randn(12, 2); y = torch.randn(12, 3); x.requires_grad_(True); y.requires_grad_(True)
    f = lambda a, b: KC.kernel_cs_native(a, b, 1.3, 0.9, chunk=5)["D_CS"]
    assert torch.autograd.gradcheck(f, (x, y), eps=1e-6, atol=1e-6)
    d0 = KC.kernel_cs_native(x.detach(), y.detach(), 1.3, 0.9, chunk=5); d1 = KC.kernel_cs_native(x.detach(), y.detach(), 1.3, 0.9, chunk=100)
    assert abs(float(d0["D_CS"] - d1["D_CS"])) < 1e-12 and d0["sigma_x_gram"] == 1.3 * math.sqrt(2)      # chunking exact; effective kernel recorded
    # closed form vs numerical integration (d = 1), and additivity over independent coordinates
    for rho in (0.3, 0.6, 0.9):
        assert abs(KC.cs_lebesgue_gaussian(rho, 1) - KC.cs_lebesgue_gaussian_quadrature(rho)) < 1e-6
        assert abs(KC.cs_lebesgue_gaussian(rho, 7) - 7 * KC.cs_lebesgue_gaussian(rho, 1)) < TOL
    # bandwidth convention, checked against the exact limit: a Gaussian KDE with bandwidth h estimates the CS divergence of the *smoothed* pair,
    # which for the bivariate normal is the same family at rho / (1 + h^2) (affine invariance of D_CS).  effective=True (Gram bandwidth sqrt(2) h)
    # is the KDE with bandwidth h; effective=False (Gram bandwidth h) is the KDE with bandwidth h / sqrt(2).  n = 3000, d = 1, rho = 0.8.
    g = torch.Generator().manual_seed(3); xs0 = torch.randn(3000, 1, generator=g); ys0 = torch.randn(3000, 1, generator=g)   # rho = 0 draw (independence: D_CS ~ 0)
    xs = torch.randn(3000, 1, generator=g); ys = 0.8 * xs + math.sqrt(1 - 0.64) * torch.randn(3000, 1, generator=g); med = KC.median_distance(xs)
    vals = []
    for mult in (0.3, 0.6):
        h = mult * med; eff = float(KC.kernel_cs_native(xs, ys, h, h, chunk=1000)["D_CS"]); raw = float(KC.kernel_cs_native(xs, ys, h, h, effective=False, chunk=1000)["D_CS"])
        assert abs(eff - KC.cs_lebesgue_gaussian(0.8 / (1 + h * h), 1)) < 0.02 and abs(raw - KC.cs_lebesgue_gaussian(0.8 / (1 + h * h / 2), 1)) < 0.02
        assert float(KC.kernel_cs_native(xs0, ys0, h, h, chunk=1000)["D_CS"]) < 0.01 < eff                       # independence ~ 0 < dependence
        vals.append(eff)
    assert vals[0] > vals[1] and vals[0] < KC.cs_lebesgue_gaussian(0.8, 1)                                         # smoothing bias shrinks with h, towards the truth


# ------------------------------------------------------------------------------------------------- Gaussian: noise rescaling, irrelevant dimensions
def test_gaussian_noise_rescaling_and_irrelevant_dimensions():
    rho, sigma = 0.6, 0.6
    cov = np.array([[1.0, rho], [rho, 1.0]]) + sigma ** 2 * np.eye(2)
    assert abs(cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1]) - rho / (1 + sigma ** 2)) < TOL
    # MC: S of the noisy pair equals S of the clean family at rho' (oracle eta of the noisy pair = eta at rho' on the standardised coordinates)
    n = 200000; g = torch.Generator().manual_seed(5); x = torch.randn(n, 1, generator=g); y = rho * x + math.sqrt(1 - rho * rho) * torch.randn(n, 1, generator=g)
    xn, yn = (x + sigma * torch.randn(n, 1, generator=g)) / math.sqrt(1 + sigma ** 2), (y + sigma * torch.randn(n, 1, generator=g)) / math.sqrt(1 + sigma ** 2)
    rp = rho / (1 + sigma ** 2); perm = torch.randperm(n, generator=g)
    S_noisy = float(0.5 * (eta(xn, yn, rp) ** 2).mean() + 0.5 * (eta(xn, yn[perm], rp) ** 2).mean())
    x2 = torch.randn(n, 1, generator=g); y2 = rp * x2 + math.sqrt(1 - rp * rp) * torch.randn(n, 1, generator=g)
    S_clean = float(0.5 * (eta(x2, y2, rp) ** 2).mean() + 0.5 * (eta(x2, y2[perm], rp) ** 2).mean())
    assert abs(S_noisy - S_clean) < 0.01                                                          # 3 predicted se ≈ 0.003
    # irrelevant dimensions: eta identical per sample, S identical, TRUTH untouched; discrete version exact (checks_v2)
    rho_s, Ds = make_gaussian(1.0, d=2, seed=0, sizes={"FIT": 64, "TUNE": 16, "SELECT": 16, "EVAL": 64, "TRUTH": 5000})
    rho_p, Dp = make_gaussian_padded(1.0, 2, 20, seed=0, sizes={"FIT": 64, "TUNE": 16, "SELECT": 16, "EVAL": 64, "TRUTH": 5000})
    assert rho_s == rho_p and Dp["EVAL"].xp.shape[1] == 20 and torch.equal(Dp["EVAL"].xp[:, :2], Ds["EVAL"].xp)
    assert torch.allclose(eta_signal(Dp["EVAL"].xp, Dp["EVAL"].yp, rho_p, 2), eta(Ds["EVAL"].xp, Ds["EVAL"].yp, rho_s))
    assert truth(1.0, 2, rho_p, Dp["TRUTH"], d_signal=2)["S_truth"] == truth(1.0, 2, rho_s, Ds["TRUTH"])["S_truth"]
    assert not torch.equal(Dp["EVAL"].xp[:, 2:5], Dp["EVAL"].xq[:, 2:5])                           # padding streams differ per side
    p, q, e = _discrete(); M = 0.5 * (p + q); S = float((M * e * e).sum()); nu = RNG.dirichlet(np.ones(6))
    pe, qe = np.outer(p, nu).ravel(), np.outer(q, nu).ravel()
    assert abs(0.5 * np.sum((pe - qe) ** 2 / (pe + qe)) - S) < TOL
    # classical CS: independent coordinates add, so padding does not change the Lebesgue value either (checked via the closed form)
    assert abs(KC.cs_lebesgue_gaussian(0.5, 2) - (KC.cs_lebesgue_gaussian(0.5, 2) + KC.cs_lebesgue_gaussian(0.0, 18))) < TOL


# ------------------------------------------------------------------------------------------------- combination: exact gain, fixed-dictionary bound, simplex
def test_combination_gain_and_dictionary_bound():
    p, q, e = _discrete(8, 6, seed=2); M = 0.5 * (p + q); m = 5
    Ts = np.tanh(RNG.normal(size=(p.size, m))); Mv = M.ravel(); pv, qv = p.ravel(), q.ravel()
    single = np.array([_J(pv, qv, Ts[:, j]) for j in range(m)])
    ub = KC.dictionary_upper_bound(Ts, weights_M=Mv)
    for _ in range(300):
        w = RNG.dirichlet(np.ones(m)); Tw = Ts @ w
        Dw = KC.dictionary_disagreement(Ts, w, weights_M=Mv)
        assert abs(_J(pv, qv, Tw) - (w @ single + Dw)) < TOL                                     # J(T_w) = sum w_j J(T_j) + D(w)
        assert _J(pv, qv, Tw) - single.max() <= Dw + TOL and Dw <= ub["U_dict"] + TOL
    # simplex QP on sample scores: weights on the simplex, objective >= best vertex
    tp = np.tanh(RNG.normal(size=(200, 3))); tn = np.tanh(RNG.normal(size=(300, 3)) - 1)
    fit = fit_small_simplex(tp, tn); w = np.asarray(fit.weights)
    assert abs(w.sum() - 1) < 1e-10 and (w >= -1e-12).all()
    assert j_hat(tp @ w, tn @ w) >= max(j_hat(tp[:, j], tn[:, j]) for j in range(3)) - 1e-10


# ------------------------------------------------------------------------------------------------- residual: pseudo-target equivalence, lambda bounds
def test_residual_pseudo_target_equivalence():
    p, q, e = _discrete(); pv, qv = p.ravel(), q.ravel(); M = 0.5 * (pv + qv)
    t0 = np.tanh(RNG.normal(size=pv.size)); u = np.tanh(RNG.normal(size=pv.size))
    for lam in (0.25, 0.5, 0.9):
        t = (1 - lam) * t0 + lam * u
        rp, rn = (1 - (1 - lam) * t0) / lam, (-1 - (1 - lam) * t0) / lam
        risk = 0.5 * (pv * (1 - t) ** 2).sum() + 0.5 * (qv * (-1 - t) ** 2).sum()
        pseudo = lam * lam * (0.5 * (pv * (rp - u) ** 2).sum() + 0.5 * (qv * (rn - u) ** 2).sum())
        assert abs(risk - pseudo) < TOL                                                            # min_U E(C - T_lam)^2 == min_U E(r - U)^2
        A = 0.5 * (pv * ((1 - t0) * (u - t0))).sum() + 0.5 * (qv * ((-1 - t0) * (u - t0))).sum(); B = (M * (u - t0) ** 2).sum()
        assert abs((_J(pv, qv, t) - _J(pv, qv, t0)) - (2 * lam * A - lam * lam * B)) < TOL
    assert abs(((1 - (1 - 0.5) * (-1.0)) / 0.5) - 3.0) < TOL and abs(((-1 - (1 - 0.5) * 1.0) / 0.5) + 3.0) < TOL   # r in [-3, 3] at lambda = 1/2


# ------------------------------------------------------------------------------------------------- noise scale: conversion and dot-product variance
def test_noise_scale_conversion_and_dot_product_variance():
    d, tau = 128, 1.0; sigma = tau / math.sqrt(d)
    assert abs(sigma * math.sqrt(d) - tau) < TOL
    z = np.zeros(d); z[0] = 1.0; w = np.zeros(d); w[0] = 0.5; w[1] = math.sqrt(0.75)
    rng = np.random.default_rng(7); n = 200000
    a = rng.normal(size=(n, d)) * sigma; b = rng.normal(size=(n, d)) * sigma
    pert = a @ w + b @ z + np.einsum("nd,nd->n", a, b)
    predicted = (2 * tau * tau + tau ** 4) / d; measured = pert.var()
    assert abs(measured - predicted) / predicted < 0.03                                            # se of a variance ≈ sqrt(2/n) ≈ 0.3 %
    assert abs(tau * math.sqrt(2 / d) - math.sqrt(2 * tau * tau / d)) < TOL                        # first-order term only
    # PCA-subspace noise: total RMS tau/sqrt(r) per coordinate inside a rank-r fixed basis
    r = 16; U = np.linalg.qr(rng.normal(size=(d, r)))[0]; nz = (rng.normal(size=(n // 10, r)) * (tau / math.sqrt(r))) @ U.T
    assert abs(np.sqrt((nz ** 2).sum(1)).mean() / tau - 1) < 0.05


# ------------------------------------------------------------------------------------------------- nested increments: oracle, finite residual, orthogonality
def test_nested_increments_oracle_and_finite_residual():
    """Binary experiment on (x, y) with 16 x-cells and 3 y-cells; A = F(x) coarsens x only (deterministic compression), a nested chain
    full -> mid -> coarse.  Oracle posteriors of the coarsenings are the M-conditional means of eta (projection), hence the increment identity
    Delta = S_B - S_A = E_M (T_B* - T_A*)^2, the orthogonality of successive oracle differences, and the finite-critic residual identity."""
    p, q, e = _discrete(16, 3, seed=4); pv, qv, ev = p.ravel(), q.ravel(), e.ravel(); M = 0.5 * (pv + qv); S = float((M * ev * ev).sum())
    xs, ys = np.divmod(np.arange(48), 3)

    def coarsen(gx):
        g = gx[xs] * 3 + ys; pp, qq = np.bincount(g, weights=pv), np.bincount(g, weights=qv); return ((pp - qq) / (pp + qq))[g]

    tm, ta = coarsen(np.arange(16) // 2), coarsen(np.arange(16) // 4)                             # nested chain: full (ev) -> mid -> coarse
    Sm, Sa = float((M * tm * tm).sum()), float((M * ta * ta).sum())
    assert S > Sm > Sa > 0.0 and abs(S - Sa) > 1e-3                                                # a real, non-degenerate experiment
    assert abs((S - Sa) - float((M * (ev - ta) ** 2).sum())) < TOL                                 # Delta = E_M (T_B* - T_A*)^2
    assert abs((Sm - Sa) - float((M * (tm - ta) ** 2).sum())) < TOL
    d1, d2 = ev - tm, tm - ta
    assert abs(float((M * d1 * d2).sum())) < TOL                                                   # oracle orthogonality -> R_orth = 0
    assert abs(float((M * d1 * d1).sum() + (M * d2 * d2).sum() - (M * (ev - ta) ** 2).sum())) < TOL   # squared increments add
    thb = np.tanh(np.arctanh(ev) + 0.1 * RNG.normal(size=48)); tha = np.tanh(np.arctanh(ta) + 0.1 * RNG.normal(size=48))
    DT = float((M * (thb - tha) ** 2).sum()); delta = S - Sa
    lhs = _J(pv, qv, thb) - _J(pv, qv, tha) - DT; rhs = 2 * float((pv * (0.5 * (1 - thb) * (thb - tha))).sum() + (qv * (0.5 * (-1 - thb) * (thb - tha))).sum())
    assert abs(lhs - rhs) < TOL                                                                    # r_BA = 2 E_M[(C - T_B)(T_B - T_A)]
    eb, ea = float((M * (thb - ev) ** 2).sum()), float((M * (tha - ta) ** 2).sum())
    assert abs(math.sqrt(DT) - math.sqrt(delta)) <= math.sqrt(eb) + math.sqrt(ea) + TOL            # |sqrt D_T - sqrt Delta| <= sqrt E_B + sqrt E_A
    # J-difference telescoping is an algebraic identity for arbitrary scores; squared-distance telescoping is not
    arb = np.tanh(RNG.normal(size=(48, 3))); js = [_J(pv, qv, arb[:, i]) for i in range(3)]
    assert abs((js[0] - js[1]) + (js[1] - js[2]) - (js[0] - js[2])) < TOL
    R_orth = float((M * ((arb[:, 0] - arb[:, 1]) ** 2 + (arb[:, 1] - arb[:, 2]) ** 2 - (arb[:, 0] - arb[:, 2]) ** 2)).sum())
    assert abs(R_orth) > 1e-6
    # log-loss increment is a Bregman (Bernoulli KL) quantity, not the squared form
    qb, qa = (1 + ev) / 2, (1 + ta) / 2; ent = lambda a: -a * np.log(a) - (1 - a) * np.log1p(-a)
    kl = qb * np.log(qb / qa) + (1 - qb) * np.log((1 - qb) / (1 - qa))
    assert abs(float((M * (ent(qa) - ent(qb) - kl)).sum())) < TOL


# ------------------------------------------------------------------------------------------------- statistical units: roles, paired-block bootstrap
def test_statistical_units_roles_and_block_bootstrap():
    uids = np.arange(45000); roles = role_split(uids); check_disjoint(roles)
    bl = eval_blocks(roles["EVAL"], seed=3)
    assert len(set(bl.ravel().tolist())) == bl.size                                                # no identity reused across blocks
    # paired-block bootstrap: resampling blocks keeps anchor and partner together (block-level indices only)
    rng = np.random.default_rng(0); idx = rng.integers(0, len(bl), len(bl)); rb = bl[idx]
    assert rb.shape == bl.shape and all(tuple(r) in {tuple(b) for b in bl} for r in rb[:50])
    # K negatives per anchor are not K independent units: the unit count is the number of anchors
    _, D = make_gaussian(1.0, d=2, seed=0, sizes={"FIT": 32, "TUNE": 8, "SELECT": 8, "EVAL": 32, "TRUTH": 64})
    assert len(D["EVAL"].xp) == 32


# ------------------------------------------------------------------------------------------------- rLS / rLS-tanh / S-Kernel / RuLSIF identity
def test_rls_rulsif_identity_and_tanh_wrap_and_rff():
    rho, D = make_gaussian(2.0, d=3, seed=1, sizes={"FIT": 600, "TUNE": 200, "SELECT": 200, "EVAL": 400, "TRUTH": 1000})
    F = D["FIT"]; rff = KC.RFFTanh(6, 64, sigma=2.0, seed=0).double()
    Fp, Fq = rff.features(F.xp, F.yp), rff.features(F.xq, F.yq)
    fit = KC.rls_fit(Fp, Fq, lam=1e-3, penalise_intercept=False)
    assert torch.allclose(fit["theta"], fit["theta_g"] - torch.eye(fit["k"], dtype=torch.float64)[-1], atol=1e-9)   # g = 1 + T (unpenalised intercept)
    gp, gq = KC.rls_apply(Fp, fit["theta_g"]), KC.rls_apply(Fq, fit["theta_g"]); tp, tq = gp - 1, gq - 1
    assert abs(KC.rulsif_ls_objective(gp, gq) - (-0.5 - 0.5 * float(j_hat(tp, tq)))) < 1e-12
    fit_pen = KC.rls_fit(Fp, Fq, lam=1e-3, penalise_intercept=True)
    assert not torch.allclose(fit_pen["theta"], fit_pen["theta_g"] - torch.eye(fit["k"], dtype=torch.float64)[-1], atol=1e-6)   # differs: recorded flag
    # raw T is unbounded in general; the wrapped version is bounded and its scalars come from a numerical fit on SELECT scores
    S = D["SELECT"]; sp, sq = KC.rls_apply(rff.features(S.xp, S.yp), fit["theta"]), KC.rls_apply(rff.features(S.xq, S.yq), fit["theta"])
    wrap = KC.fit_tanh_wrap(sp, sq); wp, wq = KC.tanh_wrap(sp, wrap["c"], wrap["b0"]), KC.tanh_wrap(sq, wrap["c"], wrap["b0"])
    assert wp.abs().max() < 1 and abs(float(j_hat(wp, wq)) - wrap["J_select"]) < 1e-9 and wrap["J_select"] >= float(j_hat(torch.tanh(sp), torch.tanh(sq))) - 1e-9
    # RFF: theta = 0 gives f = 0 (matched initial gradient scale); Nystrom features reproduce the kernel on the centres
    assert float(rff(F.xp[:5], F.yp[:5]).abs().max()) == 0.0
    C = torch.cat([F.xp[:40], F.yp[:40]], 1); ny = KC.NystromTanh(C, sigma=2.0).double()
    phi = ny.features(F.xp[:40], F.yp[:40]); Kcc = torch.exp(-KC._sqdist(C, C) / (2 * 4.0))
    assert torch.allclose(phi @ phi.T, Kcc, atol=1e-3)


# ------------------------------------------------------------------------------------------------- E-line module: estimators, oracle moments, staircase truths
def test_estimators_interface_and_oracle_values():
    st = make_setting("gaussian", 1, d=4); g = torch.Generator().manual_seed(0); x, y = st.sample(256, g)
    crit = JointMLP(4, 4).double()
    for kind in KINDS:
        for variant in ("single", "cyclic8", "inbatch"):
            fp, fn = scores(crit, x, y, variant, g); loss, val = estimate(kind, fp, fn)
            assert torch.isfinite(loss) and torch.isfinite(val) and fn.shape[1] == {"single": 1, "cyclic8": 8, "inbatch": 255}[variant]
    # VCS at the oracle critic estimates S; JS at the oracle estimates 2 JSD; both at their MC truths within 3 se on a large batch
    xb, yb = st.sample(100000, g); tr = truths_by_mc(st, 100000, seed=9)
    orc = OracleCritic(st.pmi, oracle_transform("vcs")); fp, fn = scores(orc, xb, yb, "single", g); _, S_hat = estimate("vcs", fp, fn)
    assert abs(float(S_hat) - tr["S"]) < 0.01
    orj = OracleCritic(st.pmi, oracle_transform("js")); fp, fn = scores(orj, xb, yb, "single", g); _, js_hat = estimate("js", fp, fn)
    assert abs(float(js_hat) - tr["JS2"]) < 0.01
    # oracle-table moments: E of the VCS estimate at the oracle equals S for the bivariate Gaussian (analytic S via eta on a huge draw)
    rng = np.random.default_rng(0); u, v = rng.standard_normal(2_000_000), rng.standard_normal(2_000_000)
    E, _ = estimate_moments("vcs", 0.9, u, v)
    xx = torch.randn(2_000_000, 1); yy = 0.9 * xx + math.sqrt(1 - 0.81) * torch.randn(2_000_000, 1); perm = torch.randperm(2_000_000)
    S9 = float(0.5 * (eta(xx, yy, 0.9) ** 2).mean() + 0.5 * (eta(xx, yy[perm], 0.9) ** 2).mean())
    assert abs(E - S9) < 0.003


# ------------------------------------------------------------------------------------------------- P85 runner: per-anchor decomposition, ridge moments, padded roles
def test_p85_per_anchor_decomposition_matches_native_estimates():
    g = torch.Generator().manual_seed(2); fp = torch.randn(40, generator=g); fn = torch.randn(40, 8, generator=g) - 1.0
    for kind in KINDS:
        _, val = estimate(kind, fp, fn); a, b, how = BM.per_anchor(kind, fp, fn)
        assert abs(BM.combine(a, b, how) - float(val)) < 1e-12, kind
    bt = BM.boot_units(torch.arange(10.0), torch.zeros(10), "sum", None, 50, torch.Generator().manual_seed(0))
    assert len(bt["boot_values"]) == 50 and 0 < bt["boot_se"] < 1.5
    bt2 = BM.boot_units(torch.arange(12.0), torch.zeros(12), "sum", np.repeat(np.arange(3), 4), 50, torch.Generator().manual_seed(0))
    assert set(np.round(bt2["boot_values"], 6)) <= {round(v, 6) for v in np.array([1.5, 5.5, 9.5]) @ np.array([[i, j, 3 - i - j] for i in range(4) for j in range(4 - i)]).T / 3}


def test_p85_ridge_moments_equal_full_fit_and_padded_roles():
    rho, D = make_gaussian(2.0, d=3, seed=1, sizes={"FIT": 300, "TUNE": 100, "SELECT": 100, "EVAL": 100, "TRUTH": 100}); F = D["FIT"]
    rff = KC.RFFTanh(6, 32, sigma=2.0, seed=0).double(); Fp, Fq = rff.features(F.xp, F.yp), rff.features(F.xq, F.yq)
    a, b = KC.rls_fit(Fp, Fq, 1e-2), KC.rls_from_moments(KC.rls_moments(Fp, Fq), 1e-2)
    assert torch.allclose(a["theta"], b["theta"], atol=1e-12) and torch.allclose(a["theta_g"], b["theta_g"], atol=1e-12)
    st = setting_from_mi("gaussian", 1.5, 2); roles, sizes = BM.build_roles(st, 2, 7, 64, seed=0, smoke=True)
    for r in ("FIT", "TUNE", "SELECT", "EVAL", "GRAD"):
        assert roles[r].xp.shape[1] == 7 and roles[r].xq.shape[1] == 7 and roles[r].pad_p is not None
    assert roles["TRUTH"].xp.shape[1] == 2
    E = roles["EVAL"]; x, y = st.from_base(E.base, st.param)
    assert torch.allclose(E.xp[:, :2], x.double()) and torch.allclose(E.yp[:, :2], y.double())      # base -> signal coordinates reproduce the P sample
    assert torch.equal(E.xp[:, 2:], E.pad_p[0]) and not torch.equal(E.xp[:, 2:], E.xq[:, 2:])        # padding recorded; P / Q pads independent
    tr = BM.truths(st, 2, roles["TRUTH"])
    assert abs(tr["J_oracle"] - tr["S"]) <= 3 * (tr["J_oracle_se"] + tr["S_se"]) and tr["CS_lebesgue"] == KC.cs_lebesgue_gaussian(st.param, 2)
    # xor_mixture cells never pad; unit lines round-trip; cell names are unique over the full grid
    c = PG.cell("xor_mixture", 20, 20, 4.0, 4096, 256, 2000, 0); assert c["d_signal"] == c["d_total"] == 10
    for c in PG.full()[:5] + PG.pilot():
        assert PG.parse_unit_line(PG.unit_line(c)) == c
    assert len({PG.cell_name(c) for c in PG.full()}) == len(PG.full())


# ------------------------------------------------------------------------------------------------- data authorisation (spec §12: test access is protocol-bound)
def test_test_set_authorisation_is_protocol_bound():
    """The official CIFAR test set was read once (P67/P68) and the S2 protocol (P75) has its own exemption.  P85 is synthetic-only: the
    runner records that, and no unit of the frozen grid names anything but the three synthetic generators."""
    assert BM.DATA_AUTHORISATION["official_test_accessible"] is False and BM.DATA_AUTHORISATION["image_data"] is False
    assert {c["setting"] for c in PG.full() + PG.pilot()} <= {"gaussian", "cubic", "xor_mixture"}
    assert all(c["methods"] in ("all", "vcs_kernel", "neural", "kernel") for c in PG.full() + PG.pilot())
