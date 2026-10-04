"""P134 (R1 contamination) — generators, densities, truth identities, reading helpers and a CPU smoke of one cell.  No full training."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim import benchmark as BM  # noqa: E402
from vcs_estim import r1  # noqa: E402
from vcs_estim.data import pmi_gaussian, rho_from_mi, setting_from_mi  # noqa: E402

SET = setting_from_mi("gaussian", 4.0, 20)
RHO, RHO_H = SET.param, rho_from_mi(r1.HIGH_MI, 20)


def _roles(N=512, seed=0):
    return BM.build_roles(SET, 20, 20, N, seed, smoke=True)[0]


def _C(t, e):
    return r1.Contamination(t, e, RHO, RHO_H, 20)


@pytest.mark.parametrize("ctype", r1.TYPES)
def test_eps0_is_bit_for_bit_clean(ctype):
    roles = _roles(); out, recs = r1.contaminate_roles(roles, _C(ctype, 0.0), 0)
    assert recs == {}
    for name in roles:
        for a in ("xp", "yp", "xq", "yq"):
            assert torch.equal(getattr(roles[name], a), getattr(out[name], a))
    assert BM.roles_hash(out) == BM.roles_hash(roles)


@pytest.mark.parametrize("ctype", r1.TYPES)
@pytest.mark.parametrize("eps", (0.01, 0.05, 0.1, 0.2))
def test_fraction_exact_and_indices_recorded(ctype, eps):
    roles = _roles(); role = roles["FIT"]; n = role.n
    new, rec = r1.contaminate_role(role, _C(ctype, eps), ("FIT", 0))
    k = int(round(eps * n)); assert rec["k"] == k and len(rec["p_idx"]) == k and len(set(rec["p_idx"])) == k
    changed = ((new.xp != role.xp).any(1) | (new.yp != role.yp).any(1)).nonzero().squeeze(1).tolist()
    assert sorted(changed) == rec["p_idx"]                         # exactly the recorded rows change, all others bit-for-bit equal
    qx = (new.xq != role.xq).any(1).nonzero().squeeze(1).tolist(); qy = (new.yq != role.yq).any(1).nonzero().squeeze(1).tolist()
    assert sorted(qx) == rec["qx_idx"] and sorted(qy) == rec["qy_idx"]
    exp_qx = k if ctype in ("outlier", "outlier_indep") else 0; exp_qy = k if ctype in ("outlier", "heavy") else 0
    assert len(qx) == exp_qx and len(qy) == exp_qy


def test_deterministic_and_seed_dependent():
    roles = _roles(); C = _C("heavy", 0.1)
    a, ra = r1.contaminate_role(roles["FIT"], C, ("FIT", 0)); b, rb = r1.contaminate_role(roles["FIT"], C, ("FIT", 0))
    c, rc = r1.contaminate_role(roles["FIT"], C, ("FIT", 1))
    assert torch.equal(a.yp, b.yp) and ra["p_idx"] == rb["p_idx"] and ra["p_idx"] != rc["p_idx"]


def test_independent_log_ratio_closed_form():
    g = torch.Generator().manual_seed(3); x, y = SET.sample(2000, g); x, y = x.double(), y.double()
    for e in (0.01, 0.2):
        lr = _C("independent", e).log_ratio(x, y); ref = torch.log((1 - e) * torch.exp(pmi_gaussian(x, y, RHO)) + e)
        assert torch.allclose(lr, ref, atol=1e-9)


def test_eps0_log_ratio_is_pmi():
    g = torch.Generator().manual_seed(4); x, y = SET.sample(1000, g); x, y = x.double(), y.double()
    for t in r1.TYPES:
        assert torch.allclose(_C(t, 0.0).log_ratio(x, y), pmi_gaussian(x, y, RHO), atol=1e-9)


def test_t2_density_quadrature_converged_and_normalised():
    v = torch.tensor([[0.0] * 20, [0.5] * 20, [3.0] * 20, [50.0] * 20, [1e4] + [0.0] * 19], dtype=torch.float64)
    coarse = r1.log_t2_conv(v, 0.67); fine = r1.log_t2_conv(v, 0.67, grid=(-45.0, 7.0, 12001))
    assert torch.allclose(coarse, fine, atol=1e-8, rtol=0)
    # 1-d analogue normalisation check (same quadrature, d = 1): integral of h over the line = 1
    u = torch.linspace(-400, 400, 160001, dtype=torch.float64)[:, None]; h = torch.exp(r1.log_t2_conv(u, 0.67))
    assert abs(float(torch.trapezoid(h.squeeze(), u.squeeze())) - 1.0) < 3e-3    # t_2 tail mass beyond |u| = 400 ~ 3e-6; grid error small


@pytest.mark.parametrize("ctype", r1.TYPES)
@pytest.mark.parametrize("eps", (0.05, 0.2))
def test_samplers_match_densities_bounded_identity(ctype, eps):
    """E_M[eta] = (E_P eta + E_Q eta) / 2 = 0 exactly when the samplers draw from the densities used for log r (eta bounded: reliable se)."""
    roles = BM.build_roles(SET, 20, 20, 512, 7, smoke=False)[0]                # TRUTH role: 200 000 per side
    T = roles["TRUTH"]; C = _C(ctype, eps); Tc, _ = r1.contaminate_role(T, C, ("TRUTH", 7))
    ep, eq = torch.tanh(C.log_ratio(Tc.xp, Tc.yp) / 2), torch.tanh(C.log_ratio(Tc.xq, Tc.yq) / 2)
    m = float(0.5 * ep.mean() + 0.5 * eq.mean()); se = math.sqrt(0.25 * float(ep.var()) / len(ep) + 0.25 * float(eq.var()) / len(eq))
    assert abs(m) < 4 * se, (ctype, eps, m, se)


def test_bounded_identity_detects_a_wrong_density():
    """Power check: samples drawn with the I = 10 conditional (rho_h) but log r evaluated with the I = 4 one -> the identity fails.
    (A wrong shift in outlier_indep would not do: there the contaminating pair is independent, so its log r is ~0 for any shift.)"""
    roles = BM.build_roles(SET, 20, 20, 512, 7, smoke=False)[0]; T = roles["TRUTH"]
    C = _C("outlier", 0.2); Tc, _ = r1.contaminate_role(T, C, ("TRUTH", 7))
    Cw = r1.Contamination("outlier", 0.2, RHO, rho_from_mi(4.0, 20), 20)
    ep, eq = torch.tanh(Cw.log_ratio(Tc.xp, Tc.yp) / 2), torch.tanh(Cw.log_ratio(Tc.xq, Tc.yq) / 2)
    m = float(0.5 * ep.mean() + 0.5 * eq.mean()); se = math.sqrt(0.25 * float(ep.var()) / len(ep) + 0.25 * float(eq.var()) / len(eq))
    assert abs(m) > 4 * se, (m, se)                      # rejected by the same 4-se criterion that accepts the true density


def test_contaminated_truths_eps0_limit_and_monotone_independent():
    roles = BM.build_roles(SET, 20, 20, 512, 0, smoke=True)[0]; T = roles["TRUTH"]
    clean = BM.truths(SET, 20, T); prev = clean["S"]
    for e in (0.05, 0.2):
        Tc, _ = r1.contaminate_role(T, _C("independent", e), ("TRUTH", 0)); tr = r1.contaminated_truths(_C("independent", e), Tc, clean_mi=SET.mi)
        assert tr["S"] < prev and math.isfinite(tr["MI"]) and tr["MI"] < SET.mi; prev = tr["S"]


def test_reading_helpers_toy():
    u = r1.resolution_units(0.5, 0.8, 0.9); assert u["u"] == pytest.approx(0.2) and u["u_down"] == pytest.approx(0.3) and u["u_up"] == pytest.approx(0.1)
    assert r1.normalized_shift(-0.03, 0.2, 0.1) == pytest.approx(0.15)
    assert r1.normalized_shift(-0.04, 0.2, 0.2) == pytest.approx(0.10)
    T = ("heavy", "independent", "outlier")
    assert r1.reading_label(dict.fromkeys(T, 0.5), dict.fromkeys(T, 3.0)) == "supported"
    assert r1.reading_label(dict.fromkeys(T, 0.5), dict.fromkeys(T, 0.55)) == "refuted (JS alike)"
    assert r1.reading_label(dict.fromkeys(T, 0.5), {"heavy": 0.55, "independent": 0.7, "outlier": 0.5}) == "supported"   # 0.7 > 1.2 x 0.5
    assert r1.reading_label({"heavy": 1.3, "independent": 0.2, "outlier": 0.2}, dict.fromkeys(T, 0.1)) == "not supported"


def test_cell_smoke_cpu():
    R = r1.run_r1_cell("outlier", 0.1, 0, torch.device("cpu"), smoke=True, kinds=("vcs", "js"), N=256, updates=40)
    assert R["cell"]["contamination_records"]["FIT"]["k"] == round(0.1 * 256)
    sel = [r for r in R["rows"] if r["selected"]]; assert len(sel) == 3 + 3                      # vcs x3 constructions, js x3
    assert all(math.isfinite(r["native"]["value"]) for r in sel) and math.isfinite(R["truth"]["S"])
    R0 = r1.run_r1_cell("outlier", 0.0, 0, torch.device("cpu"), smoke=True, kinds=("vcs",), N=256, updates=40)
    assert R0["cell"]["roles_hash"] == R0["cell"]["clean_roles_hash"] and R0["truth"]["MI"] == pytest.approx(4.0)
