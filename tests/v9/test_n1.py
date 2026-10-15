"""V9-N1 tests: the quantification methods on finite designs with exact truth, identification / invariance properties, the EM, BBSE, calibration,
confidence-set contract, weights and the runners' code paths (smoke-sized)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from v9 import quant as Q  # noqa: E402
from v9 import reference_core as RC  # noqa: E402

PY = sys.executable


def synth(n, kind, rng, mu=(0.3, 0.05), sd=(0.12, 0.08)):
    return rng.normal(mu[0] if kind == "P" else mu[1], sd[0] if kind == "P" else sd[1], n)


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(1)
    return Q.fit_panel(synth(600, "P", rng), synth(600, "Q", rng), synth(300, "P", rng), synth(300, "Q", rng), seed=1, adam_sensitivity=False), rng


def test_scalar_critic_positive_slope_and_logit():
    c = Q.ScalarCritic(-3.0, 0.2, "vcs")
    assert c.a > 0 and np.allclose(c.T(np.array([0.1])), np.tanh(c.a * 0.1 + 0.2)) and np.allclose(c.prob(np.array([0.1])), (1 + c.T(np.array([0.1]))) / 2)


def test_moment_affine_invariance_and_unidentified():
    a, b, c = 0.6, -0.3, 0.15
    m1 = Q.rho_moment(np.full(10, a), np.full(10, b), np.full(10, c)); m2 = Q.rho_moment(np.full(10, 2 * a + 1), np.full(10, 2 * b + 1), np.full(10, 2 * c + 1))
    assert m1["rho_raw"] == pytest.approx(m2["rho_raw"]) and m1["status"] == "point_estimate"
    assert Q.rho_moment(np.full(5, 0.3), np.full(5, 0.3), np.full(5, 0.1))["status"] == "unidentified_at_numerical_tolerance"


def test_em_recovers_prior_on_exact_posteriors():
    rng = np.random.default_rng(0); rho = 0.3; n = 20000
    s_support = np.linspace(-1, 1, 9); p = np.exp(1.2 * s_support); q = np.exp(-1.2 * s_support); p, q = p / p.sum(), q / q.sum(); post = p / (p + q)
    is_p = rng.random(n) >= rho; idx = np.where(is_p, rng.choice(9, n, p=p), rng.choice(9, n, p=q))
    e = Q.rho_em(post[idx]); assert e["status"] == "converged" and abs(e["rho_raw"] - rho) < 0.02


def test_bbse_hard_soft_and_singular():
    pp = np.array([0.9, 0.8, 0.7, 0.95]); pq = np.array([0.1, 0.2, 0.3, 0.05]); pb = np.array([0.9, 0.1, 0.8, 0.2])
    soft = Q.rho_bbse(pp, pq, pb, hard=False); hard = Q.rho_bbse(pp, pq, pb, hard=True)
    assert soft["status"] == "point_estimate" and hard["status"] == "point_estimate" and abs(soft["rho_raw"] - 0.5) < 1e-9 and abs(hard["rho_raw"] - 0.5) < 1e-9
    assert Q.rho_bbse(np.full(4, 0.6), np.full(4, 0.6), pb, hard=False)["status"] == "confusion_matrix_singular"


def test_bbse_soft_equals_bce_moment(panel):
    pan, rng = panel; sp, sq, sb = synth(100, "P", rng), synth(100, "Q", rng), np.concatenate([synth(70, "P", rng), synth(30, "Q", rng)])
    out = Q.quantify_batch(pan, sp, sq, sb)
    assert out["bbse_soft"]["rho_raw"] == pytest.approx(out["bce_moment"]["rho_raw"], abs=1e-9)   # affine in T: same estimate (documented)


def test_calibrators_fit_and_monotone():
    rng = np.random.default_rng(2); lp, lq = rng.normal(2, 1, 400), rng.normal(-2, 1, 400)
    for kind in ("ts", "bcts", "isotonic"):
        c = Q.Calibrator(kind).fit(lp, lq); p = c.prob(np.linspace(-6, 6, 50))
        assert np.all(np.diff(p) >= -1e-12) and 0 < p.min() and p.max() < 1
    assert Q.balanced_logloss(lp, lq) >= Q.balanced_logloss(*(Q.Calibrator("bcts").fit(lp, lq).params["alpha"] * x + Q.Calibrator("bcts").fit(lp, lq).params["beta"] for x in (lp, lq))) - 1e-9


def test_pava_is_isotonic_and_weighted():
    y = np.array([1.0, 0.0, 0.5, 0.2, 0.9]); w = np.ones(5); z = Q.pava(y, w)
    assert np.all(np.diff(z) >= 0) and z.sum() == pytest.approx(y.sum())


def test_all_methods_recover_rho_on_clear_design(panel):
    pan, rng = panel; rho = 0.3; n = 4000; is_p = rng.random(n) >= rho; sb = np.where(is_p, synth(n, "P", rng), synth(n, "Q", rng))
    out = Q.quantify_batch(pan, synth(2000, "P", rng), synth(2000, "Q", rng), sb, truth={"is_p": is_p, "rho": rho})
    for m in Q.METHODS:
        if m == "kernel_mpe":
            continue                                                      # conditional column (P + R only); not held to the tolerance
        assert out[m]["rho_clipped"] is not None and abs(out[m]["rho_clipped"] - out["_truth"]["rho_realized"]) < 0.06, (m, out[m])
    for m in ("vcs_moment", "bce_moment", "em_bcts", "kde_moment", "rulsif_moment", "nystrom_moment"):
        w = out[m]["weights"]; assert w["brier"] < 0.12 and w["auroc"] > 0.9 and 0 < w["ess_fraction"] <= 1
    assert "weights" not in out["raw_moment"] or out["raw_moment"].get("weights") is None


def test_weights_match_reference_posterior():
    T = np.array([0.9, -0.5, 0.0]); w = Q.weights(T, 0.3)["w"]; assert np.allclose(w, RC.clean_posterior(np.clip(T, -1 + 1e-9, 1 - 1e-9), 0.3))
    assert Q.weights(T, None)["w"] is None


def test_confidence_set_requires_disjoint_clusters():
    rp = [{"role": "P", "cluster_id": f"p{i}", "units": [f"a{i}"], "score": 0.8} for i in range(2000)]
    rq = [{"role": "Q", "cluster_id": f"q{i}", "units": [f"b{i}", f"d{i}"], "score": -0.8} for i in range(2000)]
    rr = [{"role": "R", "cluster_id": f"r{i}", "units": [f"u{i}"], "score": 0.0} for i in range(2000)]
    ok = Q.confidence_set(rp, rq, rr, fixed_score_outside_eval=True, specified_mixture=True); assert ok["status"] == "bounded_interval" and ok["lower"] <= 0.5 <= ok["upper"]
    bad = Q.confidence_set(rp, rq, rr + [{"role": "R", "cluster_id": "rX", "units": ["a0"], "score": 0.0}], fixed_score_outside_eval=True, specified_mixture=True)
    assert bad["status"] == "not_verified" and not bad["disjoint_clusters"]
    assert Q.confidence_set(rp, rq, rr, fixed_score_outside_eval=False, specified_mixture=True)["status"] == "not_verified"


def test_kernel_mpe_outputs_and_rises_with_rho():
    rng = np.random.default_rng(3); sp = synth(300, "P", rng); est = []
    for rho in (0.1, 0.5):
        n = 400; is_p = rng.random(n) >= rho; sb = np.where(is_p, synth(n, "P", rng), synth(n, "Q", rng)); r = Q.kernel_mpe(sp, sb); est.append(r["rho_clipped"])
        assert r["status"] in ("knee_estimate", "no_rise_detected") and len(r["d_curve"]) == len(r["lambdas"])
    assert est[1] > est[0]


def test_auroc_ties_and_perfect():
    assert Q.auroc(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1])) == 1.0 and Q.auroc(np.array([1, 0]), np.array([0.5, 0.5])) == 0.5


def test_exact_runner_smoke(tmp_path):
    r = subprocess.run([PY, str(ROOT / "scripts/v9/n1_exact.py"), "--out", str(tmp_path / "exact"), "--smoke", "--designs", "clear,shift", "--rhos", "0,0.3", "--repeats", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    f = tmp_path / "exact" / "clear_rho0.3_n128.json"; d = json.loads(f.read_text())
    assert d["repeats"] == 2 and d["summary"]["controls"]["constant"]["n_identified"] == 0 and d["summary"]["controls"]["affine_eta"]["mae"] == pytest.approx(d["summary"]["controls"]["oracle_eta"]["mae"], abs=1e-9)
    assert d["summary"]["controls"]["oracle_eta"]["interval"]["n"] == 2 and all(m in d["records"][0]["methods"] for m in Q.METHODS)


def test_image_runner_smoke_if_cache_present(tmp_path):
    cache = Path("/home/infres/yinwang/CS_QMI/outputs/P49_precheck_A/features/SRC-FIT.pt")
    if not cache.exists():
        pytest.skip("P49 cache not on this machine")
    r = subprocess.run([PY, str(ROOT / "scripts/v9/n1_image.py"), "--out", str(tmp_path / "image"), "--smoke", "--repeats", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    p = json.loads((tmp_path / "image" / "seed901_panel.json").read_text()); d = json.loads((tmp_path / "image" / "seed901_rho0.1_n128.json").read_text())
    assert p["sizes"]["U"] == 18000 and p["sizes"]["ref_p"] == 100 and d["summary"]["interval_vcs"]["disjoint_all"] and d["summary"]["rho_realized"]["mean"] > 0


def test_aggregate_smoke(tmp_path):
    subprocess.run([PY, str(ROOT / "scripts/v9/n1_exact.py"), "--out", str(tmp_path / "exact"), "--smoke", "--designs", "clear", "--rhos", "0,0.3", "--repeats", "2"], check=True, capture_output=True)
    r = subprocess.run([PY, str(ROOT / "scripts/v9/n1_aggregate.py"), "--exact", str(tmp_path / "exact"), "--out", str(tmp_path / "agg" / "n1")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (tmp_path / "agg" / "n1.md").exists() and (tmp_path / "agg" / "figs").exists() and json.loads((tmp_path / "agg" / "n1.json").read_text())["exact"]["n_cells"] == 2
