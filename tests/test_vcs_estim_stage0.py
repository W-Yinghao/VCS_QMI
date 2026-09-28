"""Estimator package v1, stage 0 (§5): mathematical and implementation checks.  float64 for the exact algebra; tolerances from
floating-point precision (no "1 % agreement" criteria)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim import candidates as C  # noqa: E402
from vcs_estim.bounded_core import fit_small_simplex, j_from_scores, residual_step  # noqa: E402
from vcs_estim.convex_mix import fit_js_mixture, js_mixture_logq  # noqa: E402
from vcs_estim.objectives import j_hat, js_match_loss, posterior_mse, risk_hat, vcs_loss  # noqa: E402
from vcs_estim.observation import observe  # noqa: E402
from vcs_estim.pairing import check_disjoint, eval_blocks, role_split  # noqa: E402
from vcs_estim.synthetic import eta, log_ratio, make_gaussian, rho_from_I  # noqa: E402

TOL = 1e-12
RNG = np.random.default_rng(0)


def _discrete(nx=5, ny=4):
    p = RNG.random((nx, ny)); p /= p.sum(); q = np.outer(p.sum(1), p.sum(0))
    return p, q, (p - q) / (p + q)


def test_risk_identity():
    for n_pos, n_neg in [(7, 7), (5, 40), (100, 3)]:
        tp, tn = RNG.uniform(-1, 1, n_pos), RNG.uniform(-1, 1, n_neg)
        assert abs(j_hat(tp, tn) - (1 - risk_hat(tp, tn))) < TOL
        assert abs(j_hat(tp, tn) - j_from_scores(tp, tn)) < TOL


def test_finite_gap_exact_discrete():
    p, q, e = _discrete(); M = 0.5 * (p + q); S = (M * e ** 2).sum()
    for _ in range(5):
        T = RNG.uniform(-1, 1, p.shape)
        J = (p * (T - 0.5 * T ** 2)).sum() + (q * (-T - 0.5 * T ** 2)).sum()
        assert abs((S - J) - (M * (T - e) ** 2).sum()) < TOL
    Je = (p * (e - 0.5 * e ** 2)).sum() + (q * (-e - 0.5 * e ** 2)).sum()
    assert abs(Je - S) < TOL


def test_residual_step_identity():
    for B_zero in (False, True):
        t0p, t0n = RNG.uniform(-0.9, 0.9, 50), RNG.uniform(-0.9, 0.9, 60)
        up, un = (t0p.copy(), t0n.copy()) if B_zero else (RNG.uniform(-1, 1, 50), RNG.uniform(-1, 1, 60))
        st = residual_step(t0p, t0n, up, un)
        if B_zero:
            assert st.zero_direction and st.coefficient == 0.0; continue
        for lam in (0.0, 0.3, 1.0, st.coefficient):
            dJ = j_hat(t0p + lam * (up - t0p), t0n + lam * (un - t0n)) - j_hat(t0p, t0n)
            assert abs(dJ - (2 * lam * st.A - lam ** 2 * st.B)) < TOL
        assert 0.0 <= st.coefficient <= 1.0


def test_convex_mixture_and_simplex():
    m = 3; Tp, Tn = RNG.uniform(-1, 1, (200, m)), RNG.uniform(-1, 1, (300, m))
    Tp[:, 0] = np.clip(Tp[:, 0] + 0.5, -1, 1)                       # one informative candidate
    fit = fit_small_simplex(Tp, Tn); w = fit.weights
    assert np.all(w >= 0) and abs(w.sum() - 1) < TOL
    tp, tn = Tp @ w, Tn @ w
    assert np.max(np.abs(np.r_[tp, tn])) <= 1 + TOL                  # bounded without clipping
    assert abs(fit.objective - j_hat(tp, tn)) < 1e-10                 # d'w - w'Gw = J(T_w)
    for j in range(m):
        assert fit.objective >= j_hat(Tp[:, j], Tn[:, j]) - 1e-12      # no worse than any vertex on the fit set
    assert fit.kkt_gap < 1e-8


def test_gradient_scale_at_zero():
    fp = torch.zeros(6, dtype=torch.float64, requires_grad=True); fn = torch.zeros(9, dtype=torch.float64, requires_grad=True)
    gv = torch.autograd.grad(vcs_loss(fp, fn), (fp, fn)); gj = torch.autograd.grad(js_match_loss(fp, fn), (fp, fn))
    for a, b in zip(gv, gj):
        assert torch.allclose(a, b, atol=1e-15)
    assert torch.allclose(gv[0], torch.full((6,), -1 / 6, dtype=torch.float64)) and torch.allclose(gv[1], torch.full((9,), 1 / 9, dtype=torch.float64))


def test_rulsif_identity():
    """Linear function class with an unpenalised constant: the RuLSIF (alpha = 1/2) solution g* and the closed-form VCS solution T* over the
    same features satisfy T* = g* - 1 (RuLSIF objective = -J/2 - 1/2 under T = g - 1)."""
    n_p, n_q, k = 300, 400, 6
    Pp, Pq = RNG.normal(size=(n_p, k)), RNG.normal(size=(n_q, k)) + 0.3
    Pp[:, 0] = 1; Pq[:, 0] = 1                                         # constant feature
    Gm = 0.5 * (Pp.T @ Pp / n_p + Pq.T @ Pq / n_q)
    theta_T = np.linalg.solve(2 * Gm, Pp.mean(0) - Pq.mean(0))            # max E_P T - E_Q T - E_M T^2  (J over linear T)
    theta_g = np.linalg.solve(Gm, Pp.mean(0))                             # min 1/2 E_M g^2 - E_P g   (RuLSIF, alpha = 1/2)
    e0 = np.eye(k)[0]
    assert np.allclose(theta_T, theta_g - e0, atol=1e-10)
    g = lambda P, th: P @ th
    R = lambda th: 0.5 * (0.5 * (g(Pp, th) ** 2).mean() + 0.5 * (g(Pq, th) ** 2).mean()) - g(Pp, th).mean()
    for th in (theta_g, RNG.normal(size=k)):
        T_p, T_q = g(Pp, th) - 1, g(Pq, th) - 1
        J = T_p.mean() - T_q.mean() - 0.5 * (T_p ** 2).mean() - 0.5 * (T_q ** 2).mean()
        assert abs(R(th) - (-0.5 * J - 0.5)) < TOL


def test_nested_information_exact_coarsening():
    """S(X;Y) - S(X;h(Y)) = E_M (eta - eta_h)^2 with eta_h = E_M[eta | x, h(y)] for an exact coarsening h."""
    p, q, e = _discrete(5, 6); M = 0.5 * (p + q); S = (M * e ** 2).sum()
    cells = [[0, 1], [2], [3, 4, 5]]
    ph = np.stack([p[:, c].sum(1) for c in cells], 1); qh = np.stack([q[:, c].sum(1) for c in cells], 1)
    eh = (ph - qh) / (ph + qh); Sh = (0.5 * (ph + qh) * eh ** 2).sum()
    eh_full = np.zeros_like(e)
    for j, c in enumerate(cells):
        eh_full[:, c] = eh[:, [j]]
    assert abs((S - Sh) - (M * (e - eh_full) ** 2).sum()) < TOL
    assert (qh - np.outer(ph.sum(1), ph.sum(0)) < TOL).all()             # Q_h is still the product of the coarsened marginals


def test_sample_roles_synthetic_and_images():
    _, D = make_gaussian(4.0, d=5, seed=0, sizes={"FIT": 64, "TUNE": 32, "SELECT": 32, "EVAL": 64, "TRUTH": 128})
    rows = {r: {tuple(np.round(v.xp[i].numpy(), 12)) for i in range(len(v.xp))} for r, v in D.items()}
    names = list(rows)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (rows[a] & rows[b])
    assert not torch.equal(D["FIT"].xp[:10], D["FIT"].xq[:10])            # P and Q from independent streams
    manifest = Path("/home/infres/yinwang/CS_QMI/manifests/cifar10_dev45k_val5k.json")
    if manifest.exists():
        fit_uids = json.load(open(manifest))["fit_uids"]
        roles = role_split(fit_uids); check_disjoint(roles)
        assert [len(roles[r]) for r in ("FIT", "TUNE", "SELECT", "EVAL")] == [27000, 6000, 6000, 6000]
        bl = eval_blocks(roles["EVAL"], 1)
        assert len(set(bl.ravel().tolist())) == bl.size and set(bl.ravel().tolist()) <= set(roles["EVAL"])


def test_noise_independence():
    x, y = torch.randn(20000, 3, dtype=torch.float64), torch.randn(20000, 3, dtype=torch.float64)
    gx, gy = torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)
    ox, oy = observe(x, y, 0.5, gx, gy); nx, ny = (ox - x) / 0.5, (oy - y) / 0.5
    assert abs(float((nx * ny).mean())) < 0.03 and abs(float(nx.std()) - 1) < 0.03   # independent, unit sd, no rescaling after the noise
    ox0, oy0 = observe(x, y, 0.0, gx, gy); assert torch.equal(ox0, x) and torch.equal(oy0, y)


def test_gaussian_oracle_and_candidates():
    d, I = 20, 4.0; rho = rho_from_I(I, d)
    x = torch.randn(5, d, dtype=torch.float64); y = torch.randn(5, d, dtype=torch.float64)
    # log ratio against the explicit densities
    cov = torch.eye(2 * d, dtype=torch.float64); cov[:d, d:] = rho * torch.eye(d); cov[d:, :d] = rho * torch.eye(d)
    lp = torch.distributions.MultivariateNormal(torch.zeros(2 * d, dtype=torch.float64), cov).log_prob(torch.cat([x, y], 1))
    lq = -0.5 * ((x * x).sum(1) + (y * y).sum(1)) - d * np.log(2 * np.pi)
    assert torch.allclose(log_ratio(x, y, rho), lp - lq, atol=1e-9)
    # CQ contains the oracle: set its weights to l/2
    cq = C.CQ(d).double(); r2 = rho * rho; c = 1 / (2 * (1 - r2))
    with torch.no_grad():
        cq.lin.weight[:] = torch.tensor([[rho * c, -0.5 * r2 * c, -0.5 * r2 * c]], dtype=torch.float64) * 1.0
        cq.lin.bias[:] = -0.25 * d * np.log(1 - r2)
    assert torch.allclose(torch.tanh(cq(x, y)), eta(x, y, rho), atol=1e-10)
    for fam in C.FAMILIES:
        assert C.build(fam, d)(x.float(), y.float()).shape == (5,)


def test_js_mixture_logsumexp_and_kkt():
    Fp, Fn = RNG.normal(0.5, 2, (300, 3)), RNG.normal(-0.5, 2, (300, 3)); Fp[:, 2] *= 30; Fn[:, 2] *= 30   # extreme logits: no underflow to log 0
    r = fit_js_mixture(Fp, Fn); w = np.array(r["weights"])
    lq, l1q = js_mixture_logq(Fp, w)
    assert np.isfinite(lq).all() and np.isfinite(l1q).all() and abs(w.sum() - 1) < 1e-12 and (w >= 0).all()
    assert r["kkt_gap"] < 1e-6


def test_posterior_mse_zero_at_oracle():
    ep, en = torch.rand(10, dtype=torch.float64) * 2 - 1, torch.rand(12, dtype=torch.float64) * 2 - 1
    assert float(posterior_mse(ep, en, ep, en)) == 0.0


def test_package_validator_reproduces(tmp_path):
    """The package's own validator, run on a temporary copy (it writes its report next to itself); must exit 0."""
    import shutil
    import subprocess
    src = Path(__file__).resolve().parents[1] / "estimator_package_v1" / "support"
    if not src.exists():
        pytest.skip("package copy absent")
    dst = tmp_path / "support"; shutil.copytree(src, dst)
    r = subprocess.run([sys.executable, str(dst / "validate_spec.py")], cwd=dst, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
