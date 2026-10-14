"""P156 (SEP-1) gate tests (protocol §7).  CPU only, synthetic data only."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vcs_estim import sep, sep_agg  # noqa: E402
from vcs_estim.benchmark import build_roles  # noqa: E402
from vcs_estim.data import Gaussian, pmi_gaussian, rho_from_mi, setting_from_mi  # noqa: E402
from vcs_estim.estimators import estimate  # noqa: E402

CPU = torch.device("cpu")


# ---------------------------------------------------------------- scoring rules
@pytest.mark.parametrize("arm", sep.ARMS_ALL)
@pytest.mark.parametrize("c", [1.0, -1.0])
def test_beta_gradient_closed_form(arm, c):
    """d loss / d g = (1 - T^2)^gamma (T - c) (g = f for b05 / vcs / b2; g = f / 2 for js), finite up to |g| = 20."""
    g = torch.linspace(-20, 20, 4001, dtype=torch.float64, requires_grad=True)
    f = 2 * g if arm == "js" else g
    (grad,) = torch.autograd.grad(sep.pair_loss(arm, f, c).sum(), g)
    T = torch.tanh(g).detach(); ref = (1 - T ** 2) ** sep.ARMS[arm] * (T - c)       # js: d/dg softplus(-2cg) = T - c exactly
    assert torch.isfinite(grad).all()
    assert float((grad - ref).abs().max()) < 1e-7


@pytest.mark.parametrize("arm", sep.ARMS_ALL)
def test_properness_pointwise_minimiser_is_eta(arm):
    gg = torch.linspace(-6, 6, 200001, dtype=torch.float64)
    for eta in (-0.9, -0.3, 0.0, 0.5, 0.95):
        f = 2 * gg if arm == "js" else gg
        r = (1 + eta) / 2 * sep.pair_loss(arm, f, 1.0) + (1 - eta) / 2 * sep.pair_loss(arm, f, -1.0)
        assert abs(float(torch.tanh(gg[r.argmin()])) - eta) < 1e-3


def test_js_vcs_equal_p85_losses():
    g = torch.Generator().manual_seed(0); fp = torch.randn(300, generator=g) * 3; fn = torch.randn(300, 1, generator=g) * 3
    assert torch.equal(sep.arm_loss("vcs", fp, fn), estimate("vcs", fp, fn)[0])
    assert torch.allclose(sep.arm_loss("js", fp, fn), estimate("js", fp, fn)[0], atol=0, rtol=1e-6)
    fn8 = torch.randn(300, 8, generator=g) * 3
    assert torch.equal(sep.arm_loss("vcs", fp, fn8), estimate("vcs", fp, fn8)[0])


def test_arm_T_readouts():
    f = torch.tensor([-3.0, 0.0, 2.0], dtype=torch.float64)
    assert torch.allclose(sep.arm_T("js", f), 2 * torch.sigmoid(f) - 1)
    assert torch.allclose(sep.arm_T("vcs", f), torch.tanh(f))


# ---------------------------------------------------------------- target algebra
@pytest.mark.parametrize("ctype", ["independent", "false_negative"])
@pytest.mark.parametrize("eps", [0.05, 0.1, 0.3])
def test_inverse_maps_round_trip(ctype, eps):
    eta = torch.linspace(-1, 1, 2001, dtype=torch.float64)
    back = sep.inv_map(ctype, sep.fwd_map(ctype, eta, eps), eps)
    assert float((back - eta).abs().max()) < 1e-12
    # a fitted T outside the contaminated range maps outside [-1, 1] only beyond the attainable bound
    lo = -(1 - eps) / (1 + eps)
    if ctype == "independent":
        assert float(sep.inv_map(ctype, torch.tensor(lo - 1e-3, dtype=torch.float64), eps)) < -1
        assert float(sep.inv_map(ctype, torch.tensor(lo + 1e-3, dtype=torch.float64), eps)) > -1
    assert abs(sep.amplification(0.3) - 2.4142857) < 1e-6


def test_log_ratio_closed_forms_against_densities():
    d, rho, eps = 5, 0.6, 0.3; g = torch.Generator().manual_seed(1)
    x = torch.randn(1000, d, generator=g, dtype=torch.float64); y = torch.randn(1000, d, generator=g, dtype=torch.float64)
    pmi = pmi_gaussian(x, y, rho)
    # independent: log((1 - eps) p + eps q) - log q
    assert torch.allclose(sep.log_ratio_indep(pmi, eps), torch.log((1 - eps) * torch.exp(pmi) + eps), atol=1e-10)
    # false negative: log p - log((1 - eps) q + eps p)
    assert torch.allclose(sep.log_ratio_fn(pmi, eps), pmi - torch.log((1 - eps) + eps * torch.exp(pmi)), atol=1e-10)
    # posterior maps agree with the closed-form forward maps
    eta = torch.tanh(pmi / 2)
    assert torch.allclose(torch.tanh(sep.log_ratio_indep(pmi, eps) / 2), sep.fwd_map("independent", eta, eps), atol=1e-12)
    assert torch.allclose(torch.tanh(sep.log_ratio_fn(pmi, eps) / 2), sep.fwd_map("false_negative", eta, eps), atol=1e-12)


def test_r1_independent_equals_closed_form():
    d, I, eps = 20, 4.0, 0.2; rho = rho_from_mi(I, d); P = sep.Problem("independent", eps, d, rho, CPU)
    g = torch.Generator().manual_seed(2); x = torch.randn(500, d, generator=g, dtype=torch.float64); y = torch.randn(500, d, generator=g, dtype=torch.float64)
    assert torch.allclose(P.log_ratio(x, y), sep.log_ratio_indep(pmi_gaussian(x, y, rho), eps), atol=1e-9)


def test_false_negative_sampler_density_and_power():
    """Bounded identity E_M eta = 0 on the contaminated sampler; the same identity rejects a wrong density (wrong eps)."""
    d, I, eps = 20, 4.0, 0.3; rho = rho_from_mi(I, d); s = Gaussian("gaussian", d, -1, rho, I)
    roles, _ = build_roles(s, d, d, 512, 0, smoke=True); roles["TRUTH"] = build_roles(s, d, d, 512, 0, smoke=False)[0]["TRUTH"]
    cont, recs = sep.contaminate_fn_roles(roles, eps, rho, 0)
    T = cont["TRUTH"]; assert recs["TRUTH"]["k"] == round(eps * T.n)
    tr = sep.truth_from_logr(sep.log_ratio_fn(pmi_gaussian(T.xp, T.yp, rho), eps), sep.log_ratio_fn(pmi_gaussian(T.xq, T.yq, rho), eps))
    assert tr["qc_EM_eta_ok"]
    bad = sep.truth_from_logr(sep.log_ratio_fn(pmi_gaussian(T.xp, T.yp, rho), 0.1), sep.log_ratio_fn(pmi_gaussian(T.xq, T.yq, rho), 0.1))
    assert not bad["qc_EM_eta_ok"]
    # marginals unchanged, P side untouched, exact counts in every contaminated role
    assert torch.equal(cont["TRUTH"].xp, roles["TRUTH"].xp) and torch.equal(cont["GRAD"].xq, roles["GRAD"].xq)
    for name in ("FIT", "TUNE", "SELECT", "EVAL", "TRUTH"):
        assert recs[name]["k"] == round(eps * roles[name].n)
        assert int((cont[name].xq != roles[name].xq).any(1).sum()) == recs[name]["k"]


def test_saturation_fields_direct_counts():
    g = torch.Generator().manual_seed(3); lp = torch.randn(5000, generator=g, dtype=torch.float64) * 4 + 2; lq = torch.randn(5000, generator=g, dtype=torch.float64) * 4 - 2
    tr = sep.truth_from_logr(lp, lq); ep, eq = torch.tanh(lp / 2), torch.tanh(lq / 2)
    assert abs(tr["sat95"] - (0.5 * float((ep.abs() > 0.95).double().mean()) + 0.5 * float((eq.abs() > 0.95).double().mean()))) < 1e-15
    assert abs(tr["sat95"] - tr["sat95_P_side"] - tr["sat95_Q_side"]) < 1e-12
    assert abs(tr["S"] - (0.5 * float((ep ** 2).mean()) + 0.5 * float((eq ** 2).mean()))) < 1e-15


def test_s_matching_reproduces_target():
    # production size; error budget = target MC + the bisection sample's MC (inherited by rho') + the fresh re-estimate (mean of 3 draws)
    t = sep.s_target("independent", 0.3, 20, 4.0, n=200000); rho, c0 = sep.match_rho(t["S"], 20, n=200000)
    fresh = [sep.s_clean(rho, 20, k, sep._crn_clean(20, k)) for k in (200001, 200002, 200003)]
    se = math.sqrt(t["S_se"] ** 2 + c0["S_se"] ** 2 + fresh[0]["S_se"] ** 2 / 3)
    assert abs(float(np.mean([f["S"] for f in fresh])) - t["S"]) <= 2 * se
    assert rho < rho_from_mi(4.0, 20)                                         # matched clean problem is less dependent
    t2 = sep.s_target("false_negative", 0.3, 20, 4.0, n=40000)
    assert t2["sat95_Q_side"] > 0 and t2["sat95_P_side"] == 0.0 and t["sat95_Q_side"] == 0.0   # one-sided de-saturation (T4)


# ---------------------------------------------------------------- critic classes
@pytest.mark.parametrize("arm", ["vcs", "js"])
def test_quad3_oracle_represents_clean_posterior(arm):
    d, I = 20, 4.0; rho = rho_from_mi(I, d); q = sep.Quad3(d).double()
    with torch.no_grad():
        for k, v in sep.quad3_oracle(rho, d, arm).items():
            getattr(q, k).fill_(v)
    g = torch.Generator().manual_seed(4); x = torch.randn(2000, d, generator=g, dtype=torch.float64)
    y = rho * x + math.sqrt(1 - rho * rho) * torch.randn(2000, d, generator=g, dtype=torch.float64)
    with torch.no_grad():
        T = sep.arm_T(arm, q.pairs(x, y))
    assert float((T - torch.tanh(pmi_gaussian(x, y, rho) / 2)).abs().max()) < 1e-10


def test_cos2_is_two_parameter_and_scale_free():
    c = sep.Cos2(); assert sum(p.numel() for p in c.parameters()) == 2
    with torch.no_grad():
        c.a.fill_(2.0); c.b.fill_(-0.5)
    x = torch.randn(10, 7); y = torch.randn(10, 7)
    assert torch.allclose(c.pairs(x, y), c.pairs(3 * x, 0.5 * y), atol=1e-6)
    assert c.shifted(x, y, range(1, 9)).shape == (10, 8)


def test_make_critic_p85_initialisation():
    from vcs_estim.critics import JointMLP
    torch.manual_seed(5 * 7919 + 17); ref = JointMLP(20, 20); m = sep.make_critic("mlp256", 20, 5)
    for a, b in zip(ref.parameters(), m.parameters()):
        assert torch.equal(a, b)
    assert sum(p.numel() for p in sep.make_critic("mlp8", 20, 5).parameters()) == 40 * 8 + 8 + 8 * 8 + 8 + 8 + 1


def test_cyclic_fn_injection_counts_and_law():
    d, rho, eps = 6, 0.8, 0.3; g = torch.Generator().manual_seed(6); n = 4096
    class Id:                                              # critic stub returning the pair's empirical correlation proxy x.y / d
        def pairs(self, x, y):
            return (x * y).sum(-1) / x.shape[1]
    x = torch.randn(n, d, generator=g, dtype=torch.float64); fn = torch.zeros(n, 8, dtype=torch.float64)
    out, k = sep.inject_cyclic_fn(Id(), x, fn, eps, rho, g)
    assert k == round(eps * n * 8) and int((out != 0).sum()) == k
    assert abs(float(out[out != 0].mean()) - rho) < 0.02    # replaced negatives are dependent pairs (E x.y / d = rho)


# ---------------------------------------------------------------- roles, determinism, cells
@pytest.mark.parametrize("d,I", [(20, 2.0), (20, 8.0), (64, 4.0)])
def test_eps0_roles_bit_for_bit_p85(d, I):
    s85 = setting_from_mi("gaussian", I, d); r85, _ = build_roles(s85, d, d, 256, 100, smoke=True)
    s = Gaussian("gaussian", d, -1, rho_from_mi(I, d), I); r, _ = build_roles(s, d, d, 256, 100, smoke=True)
    for k in r85:
        for a in ("xp", "yp", "xq", "yq"):
            assert torch.equal(getattr(r85[k], a), getattr(r[k], a))


def test_cells_for_seed_counts_and_names_unique():
    cs = sep.cells_for_seed(100); from collections import Counter
    assert dict(Counter(c["block"] for c in cs)) == {"A": 54, "E": 36, "C": 14, "D": 12, "Dp": 7, "B": 24, "B2": 12}
    names = [sep.cell_name(c) for c in cs]; assert len(set(names)) == len(names)


def _tiny(cell_kw, smoke=True):
    c = sep.cell(**cell_kw); return sep.run_cell(c, CPU, smoke=smoke)


def test_tiny_cell_end_to_end_and_deterministic():
    kw = dict(block="A", cls="mlp256", variant="product", ctype="independent", eps=0.3, d=20, I=4.0, N=256, U=40, arms=sep.ARMS_ALL, lrs=sep.LRS_MLP, seed=100)
    R1, R2 = _tiny(kw), _tiny(kw)
    e1 = [r["readout"]["posterior_mse"] for r in R1["rows"]]; e2 = [r["readout"]["posterior_mse"] for r in R2["rows"]]
    assert e1 == e2 and all(math.isfinite(v) for v in e1)
    assert {r["arm"] for r in R1["rows"] if r["selected"]} == set(sep.ARMS_ALL)
    assert all("clean_target" in r["readout"] for r in R1["rows"])
    s = sum(v["contribution"] for v in R1["rows"][0]["readout"]["regions"].values())
    assert abs(s - R1["rows"][0]["readout"]["posterior_mse"]) < 1e-12


@pytest.mark.parametrize("kw", [
    dict(block="Dp", cls="mlp256", variant="cyclic8", ctype="false_negative", eps=0.3, d=20, I=4.0, N=256, U=40, arms=sep.ARMS_PAIR, lrs=sep.LRS_MLP, seed=100),
    dict(block="E", cls="cos2", variant="product", ctype="independent", eps=0.1, d=20, I=4.0, N=256, U=40, arms=sep.ARMS_ALL, lrs=sep.LRS_PAR, seed=100),
    dict(block="E", cls="quad3", variant="product", ctype="clean", eps=0.0, d=64, I=8.0, N=256, U=40, arms=sep.ARMS_ALL, lrs=sep.LRS_PAR, seed=100),
    dict(block="D", cls="mlp256", variant="product", ctype="heavy", eps=0.2, d=20, I=4.0, N=256, U=40, arms=sep.ARMS_PAIR, lrs=sep.LRS_MLP, seed=100),
])
def test_tiny_cells_other_blocks(kw):
    R = _tiny(kw); assert all(math.isfinite(r["readout"]["posterior_mse"]) for r in R["rows"])
    if kw["variant"] == "cyclic8":
        assert all(r["select"]["cyclic_fn_injected"] == 40 * round(0.3 * 256 * 8) for r in R["rows"])


def test_saturation_matching():
    t = sep.s_target("independent", 0.3, 20, 4.0, n=200000); rs, c = sep.match_rho_sat(t["sat95"], 20, n=200000)
    assert abs(c["sat95"] - t["sat95"]) < 2e-4                                 # bisection on a step function of rho: within ~1 / n of the goal
    fresh = sep.s_clean(rs, 20, 200001, sep._crn_clean(20, 200001))
    assert abs(fresh["sat95"] - t["sat95"]) < 0.005


def test_matched_sat_cell_uses_rho_sat():
    row = {"ctype": "independent", "eps": 0.3, "d": 20, "I": 4.0, "rho_prime": 0.4, "rho_sat": 0.45}
    c = sep.cell("B2", "mlp256", "product", "matched", 0.0, 20, 4.0, 256, 40, sep.ARMS_PAIR, sep.LRS_MLP, 100, match=("independent", 0.3, "sat95"))
    R = sep.run_cell(c, CPU, smoke=True, matching={("independent", 0.3, 20, 4.0): row})
    cS = sep.cell("B", "mlp256", "product", "matched", 0.0, 20, 4.0, 256, 40, sep.ARMS_PAIR, sep.LRS_MLP, 100, match=("independent", 0.3))
    assert abs(R["cell"]["rho"] - 0.45) < 1e-12 and "0.3sat" in sep.cell_name(c) and sep.cell_name(c) != sep.cell_name(cS)


def test_matched_cell_uses_table(tmp_path):
    row = {"ctype": "independent", "eps": 0.3, "d": 20, "I": 4.0, "rho_prime": 0.4, "rho_sat": 0.45}
    c = sep.cell("B", "mlp256", "product", "matched", 0.0, 20, 4.0, 256, 40, sep.ARMS_PAIR, sep.LRS_MLP, 100, match=("independent", 0.3))
    R = sep.run_cell(c, CPU, smoke=True, matching={("independent", 0.3, 20, 4.0): row})
    assert abs(R["cell"]["rho"] - 0.4) < 1e-12 and R["cell"]["match_row"] == row


# ---------------------------------------------------------------- aggregate on toy cells with known effects
def _fake(block, cls, ctype, eps, d, I, seed, e_vcs, e_js, match=None, N=4096, U=2000, variant="product"):
    c = sep.cell(block, cls, variant, ctype, eps, d, I, N, U, sep.ARMS_PAIR, sep.LRS_MLP, seed, match=match)
    rows = [{"arm": a, "selected": True, "selected_commonJ": True, "lr": 1e-4, "lr_edge": None, "select": {"not_converged": False},
             "readout": {"posterior_mse": e}, "readout_commonJ_selected": {"posterior_mse": e}} for a, e in (("vcs", e_vcs), ("js", e_js))]
    return {"cell": {**c, "name": sep.cell_name(c), "smoke": True}, "truth": {"S": 0.5, "sat95": 0.3, "sat95_P_side": 0.1, "sat95_Q_side": 0.2}, "rows": rows}


def test_aggregate_known_effects(tmp_path):
    rng = np.random.default_rng(0)
    for s in range(100, 108):
        for ct in ("independent", "false_negative"):
            for d in (20, 64):
                for I in (2.0, 4.0, 8.0):
                    noise = rng.normal(0, 1e-4)
                    for f in [_fake("A", "mlp256", "clean", 0.0, d, I, s, 0.11, 0.10),                       # clean gap +0.01 (JS better)
                              _fake("A", "mlp256", ct, 0.3, d, I, s, 0.10 + noise, 0.10),                     # contaminated gap 0 -> G = -0.1 (scale 0.1)
                              _fake("B", "mlp256", "matched", 0.0, d, I, s, 0.10 + noise, 0.10, match=(ct, 0.3)),   # matched gap 0 -> H = 0
                              _fake("B2", "mlp256", "matched", 0.0, d, I, s, 0.12, 0.10, match=(ct, 0.3, "sat95"))]:  # sat-matched gap +0.02 -> H_sat = -0.2
                        (tmp_path / (f["cell"]["name"] + ".json")).write_text(json.dumps(f))
        for cls in ("cos2", "mlp8"):
            for d in (20, 64):
                for I in (4.0, 8.0):
                    f = _fake("E", cls, "clean", 0.0, d, I, s, 0.18 + rng.normal(0, 1e-4), 0.20)                # restricted: VCS better by 0.02 / 0.2 = -0.1
                    (tmp_path / (f["cell"]["name"] + ".json")).write_text(json.dumps(f))
    cells, smoke = sep_agg.load(tmp_path); res = sep_agg.aggregate(cells); P = res["primary"]
    assert smoke and P["P1_contamination_narrows_gap"]["stat"]["n"] == 8
    assert abs(P["P1_contamination_narrows_gap"]["stat"]["mean"] + 0.1) < 1e-2 and P["P1_contamination_narrows_gap"]["reading"].startswith("<")
    assert P["P2_contamination_specific"]["reading"] == "n.s." or abs(P["P2_contamination_specific"]["stat"]["mean"]) < 1e-2
    assert abs(P["P3_restricted_class_reversal"]["stat"]["mean"] + 0.1) < 1e-2 and P["P3_restricted_class_reversal"]["reading"].startswith("<")
    assert any(c.startswith("O1") for c in res["outcome_candidates"]) and any(c.startswith("O4") for c in res["outcome_candidates"])
    assert abs(res["secondary"]["P2_sat_pooled"]["stat"]["mean"] + 0.2) < 1e-2
    assert sep_agg.markdown(res, smoke).startswith("# P156")


def test_holm():
    adj = sep_agg.holm([0.01, 0.04, 0.03]); assert adj == pytest.approx([0.03, 0.06, 0.06])
