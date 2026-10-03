"""P116 (v5 NEXT-E-SOLVE) tests: backward compatibility of the edited fitters, shared splits, solver residual fields, the A2 step-0 property,
vectorised permutation tests, and an exact reproduction of a historical P105 instance."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import precheck_d_tests as PD  # noqa: E402
import cond_test_t1_ablation as AB  # noqa: E402
from vcs_measure import solve as S  # noqa: E402

torch.set_default_dtype(torch.float32)


def _data(n=300, d=16, seed=0, strength=0.6):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g)
    nb = (torch.rand(n, generator=g) < 0.5).long()
    z[:, 0] += strength * (2 * nb.float() - 1)           # planted dependence on coordinate 0
    nneg = (torch.rand(n, generator=g) < 0.5).long()
    return z, nb, nneg


def _old_split(n, seed, frac=0.2):
    g = torch.Generator().manual_seed(seed); perm = torch.randperm(n, generator=g); nv = max(8, int(frac * n))
    return perm[nv:], perm[:nv]


def test_closed_form_default_equals_explicit_split_and_A1():
    z, n, nn = _data()
    old = PD.closed_form_critic(z, n, nn, 7)
    ex = PD.closed_form_critic(z, n, nn, 7, split=_old_split(len(z), 7))
    assert torch.equal(old.w, ex.w) and old.c == ex.c and old.lam == ex.lam and old.val_J == ex.val_J
    assert PD.ridge_tanh_calibrated is PD.closed_form_critic and old.solver["name"] == "ridge_tanh_calibrated"
    ti, vi = S.make_split(len(z), 7)
    a1 = S.ridge_tanh_calibrated(S.Design.build(z, n, nn, ti, vi))
    assert a1.meta["lam"] == old.lam and a1.meta["c"] == old.c and abs(a1.meta["val_J"] - old.val_J) < 1e-12
    T_old = torch.tanh(old(z, n).double()); T_new = a1.T(z, n)
    assert float((T_old - T_new).abs().max()) < 1e-5  # CF returns a float32 pre-activation (atanh of clamped tanh)


def test_fit_vcs_and_c2st_default_equals_explicit_split():
    z, n, nn = _data()
    for kind in ("linear", "mlp"):
        a = PD.fit_vcs_critic(z, n, nn, 30, 3, kind=kind)
        b = PD.fit_vcs_critic(z, n, nn, 30, 3, kind=kind, split=_old_split(len(z), 3))
        assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()) and a.val_J == b.val_J
    a = PD.fit_c2st(z, n, nn, 30, 5); b = PD.fit_c2st(z, n, nn, 30, 5, split=_old_split(len(z), 5))
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())


def test_exact_js_default_equals_split_and_residual_fields():
    z, n, nn = _data()
    a = AB.exact_js_critic(z, n, nn, 11)
    b = AB.exact_js_critic(z, n, nn, 11, split=_old_split(len(z), 11))
    assert torch.equal(a.w, b.w) and a.lam == b.lam and a.val_J == b.val_J
    for d in a.solver["per_lam"]:
        for k in ("n_iter", "func_evals", "termination", "final_grad_maxabs", "final_grad_l2", "last_objective_change", "max_iter"):
            assert k in d
        assert d["termination"] in ("tolerance_grad", "max_iter", "max_eval", "tolerance_change")
    assert a.solver["chosen"]["lam"] == a.lam


def test_shared_split_identical_across_methods():
    z, n, nn = _data()
    ti, vi = S.make_split(len(z), 42)
    D = S.Design.build(z, n, nn, ti, vi)
    a3 = S.js_lbfgs(D, budgets=(20, 40))
    ref = AB.exact_js_critic(z, n, nn, 999, split=(ti, vi))                    # init seed differs, split shared
    # both VAL scores are computed on the same VAL rows: recompute from the weights on D
    assert abs(float(S.js_value(D.vp @ ref.w, D.vn @ ref.w)) - ref.val_J) < 1e-5
    assert abs(float(S.js_value(D.vp @ a3[40].v, D.vn @ a3[40].v)) - a3[40].meta["val_JS"]) < 1e-12
    vcs = PD.closed_form_critic(z, n, nn, 123, split=(ti, vi))
    assert abs(vcs.val_J - S.ridge_tanh_calibrated(D).meta["val_J"]) < 1e-12


def test_A2_step0_is_A1_and_never_worse_on_fit_or_val():
    z, n, nn = _data(seed=3)
    ti, vi = S.make_split(len(z), 5); D = S.Design.build(z, n, nn, ti, vi)
    a1 = S.ridge_tanh_calibrated(D)
    out = S.ridge_then_bounded_j(D, a1, budgets=(40, 120, 240))
    fit_a1 = S.risks(a1, D)["own_risk_fit"]
    sc = D.sc
    assert {("last", B) for B in (40, 120, 240)} <= set(out)
    for B, c in ((k, v) for k, v in out.items() if not isinstance(k, tuple)):
        assert abs(c.meta["fit_obj_step0"] - fit_a1) < 1e-12                  # step 0 == A1 on FIT (unpenalised)
        assert c.meta["val_J"] >= a1.meta["val_J"] - 1e-12                     # A1 is a candidate: VAL J never below A1's
        for r in c.meta["per_lam"]:                                            # strong-Wolfe L-BFGS never increases the PENALISED FIT objective
            pen0 = fit_a1 + 0.5 * r["lam"] * sc * float((a1.v ** 2).sum())
            assert r["residual_last"]["fit_objective"] <= pen0 + 1e-12
        for k in ("grad_maxabs", "grad_l2", "fit_objective"):
            assert k in c.meta["residual_chosen"]


def test_perm_tests_match_loop():
    z, n, nn = _data(n=120)
    ti, vi = S.make_split(len(z), 1); D = S.Design.build(z, n, nn, ti, vi)
    for cr in (S.ridge_tanh_calibrated(D), S.js_lbfgs(D, budgets=(40,))[40]):
        g = torch.Generator().manual_seed(0)
        PN = torch.stack([n[torch.randperm(len(n), generator=g)] for _ in range(30)])
        pt = S.perm_tests(cr, z, n, nn, PN, 0.05)
        Tn = cr.T(z, nn)
        def J(Tp):
            return float((Tp - 0.5 * Tp ** 2).mean() + (-Tn - 0.5 * Tn ** 2).mean())
        obs = J(cr.T(z, n)); null = [J(cr.T(z, pn)) for pn in PN]
        p = (1 + sum(x >= obs for x in null)) / (1 + len(null))
        assert abs(pt["common"]["stat"] - obs) < 1e-10 and abs(pt["common"]["p"] - p) < 1e-12


SMOKE = REPO / "reports" / "P105_smoke" / "smoke.json"
FEAT = Path("/home/infres/yinwang/CS_QMI/outputs/P45_precheck_D_vcs4v800")
PRE_EDIT = {"precheck_d_tests": "df6345d", "cond_test_t1_ablation": "d98452a"}   # last commits before the P116 edits


def _load_pre_edit(tmp_path):
    """Write the pre-edit versions of the two edited scripts (from git) into a temp dir and import them as separate modules."""
    import importlib.util, subprocess
    mods = {}
    for name, rev in PRE_EDIT.items():
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:scripts/{name}.py"], capture_output=True, text=True, check=True).stdout
        (tmp_path / f"{name}.py").write_text(src)
    sys.path.insert(0, str(tmp_path))
    try:
        for name in ("precheck_d_tests", "cond_test_t1_ablation"):
            spec = importlib.util.spec_from_file_location(f"pre_{name}", tmp_path / f"{name}.py")
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); mods[name] = m
    finally:
        sys.path.remove(str(tmp_path))
    return mods


@pytest.mark.skipif(not FEAT.is_dir(), reason="features absent")
def test_new_code_bit_identical_to_pre_edit_code(tmp_path):
    """Same node, same process: the edited fitters with default arguments give bit-identical P105 instances to the pre-edit code."""
    if DEVICE_IS_CUDA:
        pytest.skip("CPU check")
    torch.set_num_threads(8)
    from cond_test_t1 import Family
    fam = Family(FEAT)
    pre = _load_pre_edit(tmp_path)
    a = AB.run_repeat(fam, 200, np.random.default_rng(999), mode="planted", strength=0.2, delta=0.05, perms=20, steps=60, seed=999000)
    b = pre["cond_test_t1_ablation"].run_repeat(fam, 200, np.random.default_rng(999), mode="planted", strength=0.2, delta=0.05, perms=20, steps=60, seed=999000)
    for k in AB.TESTS:
        assert a[k]["stat"] == b[k]["stat"] and a[k]["p"] == b[k]["p"] and a[k]["val"] == b[k]["val"], k


@pytest.mark.skipif(not (SMOKE.is_file() and FEAT.is_dir()), reason="historical smoke or features absent")
def test_reproduces_historical_P105_smoke_instance():
    """Default arguments reproduce instance 0 of the P105 CPU smoke (job 1016884 on nodecpu05; colour s0.2, n 200, 20 perms, 60 steps, seed 999):
    identical p-values for every critic; statistics equal to rel 1e-5 (the float32 MLP critics drift at ~3e-6 relative across CPU nodes; the
    linear / ridge / L-BFGS critics match to float precision).  Bit-identity of new vs pre-edit code is the test above."""
    if DEVICE_IS_CUDA:
        pytest.skip("historical smoke ran on CPU")
    torch.set_num_threads(8)
    from cond_test_t1 import Family
    fam = Family(FEAT)
    rng = np.random.default_rng(999)
    inst = AB.run_repeat(fam, 200, rng, mode="planted", strength=0.2, delta=0.05, perms=20, steps=60, seed=999 * 1000 + 0)
    hist = json.loads(SMOKE.read_text())["cases"]["colour_s0.2"]["by_n"]["200"]["instances"][0]
    for k in AB.FORCED:
        assert inst[k]["p"] == hist[k]["p"], k
        assert math.isclose(inst[k]["stat"], hist[k]["stat"], rel_tol=1e-5, abs_tol=1e-9), (k, inst[k]["stat"], hist[k]["stat"])
        assert math.isclose(inst[k]["val"], hist[k]["val"], rel_tol=1e-5, abs_tol=1e-9), k
    for k in ("vcs_closed", "js_exact"):
        assert math.isclose(inst[k]["stat"], hist[k]["stat"], rel_tol=1e-9, abs_tol=1e-12), k


DEVICE_IS_CUDA = PD.DEVICE.type == "cuda"
