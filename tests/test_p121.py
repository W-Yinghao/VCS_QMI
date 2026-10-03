"""P121 (v6 V6-THEORY) tests: exact identities on the finite latent model, nested decomposition, Gram fit bookkeeping, gradient formulas.
Also re-runs the v6 package's own 27 reference checks against the copied core (src/vcs_theory/geometry_evidence_core.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from vcs_theory import p121  # noqa: E402
from vcs_theory import geometry_evidence_core as core  # noqa: E402


def test_latent_gamma1_reproduces_core():
    d, c = p121.latent(seed=73, gamma=1.0), core.latent_model(seed=73)
    np.testing.assert_allclose(d["P"], c["P"], atol=1e-15); assert abs(d["S"] - c["S"]) < 1e-15


@pytest.mark.parametrize("name", list(p121.CONDITIONS))
def test_kernel_identity_and_normalisation(name):
    d = p121.latent(**p121.CONDITIONS[name]); k = p121.kernel_identity(d)
    assert k["pass"] and abs(k["global_normalisation"] - 1) < 1e-12 and k["max_anchor_normalisation_error"] < 1e-12
    assert (d["P"] > 0).all() and (d["Q"] > 0).all() and 0 < d["S"] < 1


@pytest.mark.parametrize("name", list(p121.CONDITIONS))
def test_nested_decomposition_exact(name):
    d = p121.latent(**p121.CONDITIONS[name]); out = p121.nested_check(d, n_draws=20)
    assert out["pass"] and out["data_processing"]["abs_diff"] < 1e-14


def test_dependence_increases_with_gamma():
    S = [p121.latent(seed=11, gamma=g)["S"] for g in (0.5, 2.0, 4.0)]
    assert S[0] < S[1] < S[2]


def test_unit_vector_fit_bookkeeping():
    d = p121.latent(seed=11, gamma=2.0); r = p121.fit_unit_vectors(d, 2.0, 0.5, 4, 0, steps=100)
    assert r["J"] <= d["S"] + 1e-12 and abs(r["gap_S_minus_J"] - r["posterior_mse"]) < 1e-12 and r["lipschitz_holds"]
    assert r["J_trace"][-1][1] >= r["J_trace"][0][1] - 1e-9


def test_gradient_checks_pass():
    g = p121.gradient_checks(); assert g["pass"], {k: v for k, v in g.items() if k.endswith("err")}
    assert g["s_equals_1_grad_norm"] < 1e-12 < g["s_equals_1_scalar_gate"]
    assert abs(g["js_vs_vcs_at_f0"]["c=+1"]["js"] - g["js_vs_vcs_at_f0"]["c=+1"]["vcs"]) < 1e-15
    for c in g["curve"]:
        assert abs(c["f0"] + 1.0) < 1e-15 and abs(c["f1"] - 1.0) < 1e-15 and 0.25 - 1e-12 <= c["min_slope_over_a"] and c["max_slope_over_a"] <= 1.75 + 1e-12
    zero = {c["curvature"]: c["actual_zero"] for c in g["curve"]}
    assert abs(zero[0.0] - 0.5) < 1e-12 and abs(zero[0.25] - 0.5) > 1e-3 and abs(zero[-0.25] - 0.5) > 1e-3


def test_package_reference_checks():
    import importlib.util, unittest
    S = REPO.parent / "VCS_Server_Tasks_and_Theory_v6_20261003.zip"
    spec_dir = Path("/tmp/claude-34987/-home-infres-yinwang-CS-QMI/01a10c49-45e0-40ba-9c3d-3d46a1404012/scratchpad/v6/VCS_Server_v6/support")
    if not (spec_dir / "test_geometry_evidence_core.py").is_file():
        pytest.skip("v6 support tests not unpacked")
    sys.modules["geometry_evidence_core"] = core            # run the package's tests against the COPIED core
    spec = importlib.util.spec_from_file_location("v6_ref_tests", spec_dir / "test_geometry_evidence_core.py"); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = unittest.TextTestRunner(verbosity=0).run(unittest.defaultTestLoader.loadTestsFromModule(mod))
    assert res.wasSuccessful() and res.testsRun == 27 and not res.skipped
