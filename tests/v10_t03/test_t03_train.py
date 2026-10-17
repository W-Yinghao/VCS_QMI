"""Tiny end-to-end trainings of the five families (a few steps on a tiny suite): optimiser groups, W rules, the M1 relative zero-head
rule as the adapter implements it, shared streams / initial states, determinism, checkpoints, own-CAL selection and the evaluator."""
from __future__ import annotations

import json

import pytest
import torch

import t03_common as C
import t03_data as D
import t03_models as Mo
import t03_objectives as O
import t03_run as R
import t03_train as TR
from v10 import adapter as A

CASE = ("diagonal_mixture", 8, 0.75)
METHODS = (("MF_VCS_perm1", 4, None), ("MF_VCS_M0", 4, None), ("MF_VCS_M1", 4, 10), ("MLP_VCS_perm1", None, None), ("MLP_VCS_perm8", None, None))


def _cfg(method, K, cap, uid=None, steps=4, ckpt=2, lr=1e-3, algo=1017):
    return TR.UnitConfig(unit_id=uid or f"T_{method}_{cap}", block="smoke", method=method, family=CASE[0], m=CASE[1], rho=CASE[2], K=K,
                         lr=lr, cap=cap, data_root_seed=7, algo_seed=algo, steps=steps, ckpt_every=ckpt)


def _run(cfg, suite, record, out_root):
    ident = {"t03_code_hash": "test"}
    res = R.execute_unit(cfg, suite=suite, record=record, out=R.unit_dir(cfg.unit_id, out_root), ident=ident, env={}, extra={})
    assert res["status"] == "COMPLETE", res.get("failure")
    return res


@pytest.fixture()
def five(tiny_suite, out_root):
    suite, record = tiny_suite
    return {m: _run(_cfg(m, K, cap), suite, record, out_root) for m, K, cap in METHODS}, suite, record, out_root


def test_all_families_complete_with_artifacts(five):
    res, suite, record, out_root = five
    for meth, r in res.items():
        d = R.unit_dir(r["unit_id"], out_root)
        assert [c["step"] for c in r["checkpoints"]] == [2, 4]
        assert (d / "checkpoints.pt").exists() and (d / "train_steps.json.gz").exists()
        ev = json.loads((d / "evaluator_only" / "evaluation.json").read_text())
        assert ev["status"] == "PASS" and r["evaluation"]["sha256"] == C.sha256_file(d / "evaluator_only" / "evaluation.json")
        sm = ev["selected_metrics"]
        assert abs(sm["identity_residual"]) <= C.IDENTITY_ATOL
        assert ev["selected"]["step"] == r["selection"]["own_cal"]["step"]
        assert sm["head_gap"]["status"] == ("MEASURED" if meth.startswith("MF_") else "NOT_APPLICABLE")
        if meth.startswith("MF_"):
            hp = sm["head_gap"]["fp64_polish"]
            assert hp["status"] in ("gap_tolerance", "iteration_budget_exhausted") and hp["suboptimality_lower_bound"] >= -1e-12
            assert hp["suboptimality_lower_bound"] <= hp["suboptimality_upper_bound_fw_gap"] + 1e-12   # J* - J(W) <= FW gap(W)
        assert set(sm["perm_conditional_variance"]) == {"fit", "eval"}
        assert ev["readout_reproduction"]["exact"] is True
        assert r["truth_barrier"]["denials_during_unit"] == []
        assert r["data"]["generator_sha256"] == C.GENERATOR_SHA256 and r["data"]["sample_sha256"]["fit"] == record["cases"][r["case_id"]]["roles"]["fit"]["sample_sha256"]


def test_optimiser_groups_and_w_rules(five):
    res = five[0]
    g = {m: {x["name"]: x for x in r["optimizer"]["groups"]} for m, r in res.items()}
    for m in ("MF_VCS_perm1", "MF_VCS_M0"):
        assert set(g[m]) == {"feature_maps", "W"} and g[m]["W"]["weight_decay"] == 0.0 and g[m]["feature_maps"]["weight_decay"] == 1e-4
        assert g[m]["W"]["lr"] == g[m]["feature_maps"]["lr"] == 1e-3
    assert set(g["MF_VCS_M1"]) == {"feature_maps"}
    assert set(g["MLP_VCS_perm1"]) == {"mlp_critic"} and g["MLP_VCS_perm1"]["mlp_critic"]["weight_decay"] == 1e-4
    for r in res.values():
        assert r["optimizer"]["clip_norm"] == 1.0 and r["optimizer"]["defaults"]["betas"] == (0.9, 0.999)
        assert all(c["w_absmax"] is None or c["w_absmax"] <= 1.0 for c in r["checkpoints"])


def test_shared_init_and_streams(five):
    res = five[0]
    assert len({res[m]["critic"]["init_state_sha256"] for m in ("MF_VCS_perm1", "MF_VCS_M0", "MF_VCS_M1")}) == 1
    assert res["MLP_VCS_perm1"]["critic"]["init_state_sha256"] == res["MLP_VCS_perm8"]["critic"]["init_state_sha256"]
    assert res["MF_VCS_perm1"]["streams"]["perm"] == res["MLP_VCS_perm1"]["streams"]["perm"] == res["MLP_VCS_perm8"]["streams"]["perm"]
    d8 = res["MLP_VCS_perm8"]["derangements"]
    assert d8["perm"]["draws"] == 4 and d8["perm8_extra"]["draws"] == 7 * 4
    assert res["MF_VCS_M0"]["streams"]["perm"] is None and res["MF_VCS_M1"]["derangements"]["perm"] is None


def test_m1_log_follows_the_relative_zero_head_rule(five):
    res, suite, record, out_root = five
    r = res["MF_VCS_M1"]
    log = C.read_json(R.unit_dir(r["unit_id"], out_root) / "train_steps.json.gz")
    assert log["m1_W_start_is_zero"][0] is True and log["m1_W_start_is_zero"][1] is False     # W = 0 first, warm start afterwards
    for g, tau, it, rs, mp in zip(log["m1_gap_ref_zero"], log["m1_tolerance_effective"], log["m1_iterations"], log["m1_stop_reason"],
                                  log["m1_moments_passes"]):
        assert tau == pytest.approx(0.01 * g, rel=1e-15) and 0 <= it <= 10 and mp == 2
        assert rs in ("gap_tolerance", "iteration_budget_exhausted", "zero_reference_gap")
    assert log["m1_gap_start"][0] == log["m1_gap_ref_zero"][0]                               # from W = 0 the start gap is G(0)
    assert r["m1"]["rule"] == "relative_zero_head" and r["m1"]["require_converged"] is False and r["m1"]["cap"] == 10
    for h, hd, ht, mp, md, mt, trig in zip(log["m1_hvp_calls"], log["m1_fp64_diag_hvp_calls"], log["m1_hvp_calls_total"], log["m1_moments_passes"],
                                           log["m1_fp64_diag_moments_passes"], log["m1_moments_passes_total"], log["m1_fp64_triggered"]):
        assert ht == h + hd and mt == mp + md and (md == 1) == trig          # decision 2.3: the fp64 diagnostic's work is counted
    assert r["m1"]["hvp_calls_total_sum"] == sum(log["m1_hvp_calls_total"]) >= r["m1"]["hvp_calls_sum"]
    assert r["m1"]["moments_passes_total_sum"] == r["m1"]["moments_passes_sum"] + r["m1"]["fp64_diag_moments_passes_sum"]


def test_m1_step_equals_the_adapter_rule_on_the_same_tensors(tiny_suite, out_root):
    suite, record = tiny_suite
    cfg = _cfg("MF_VCS_M1", 4, 30, uid="M1_ONE", steps=1, ckpt=1)
    res = _run(cfg, suite, record, out_root)
    blob = torch.load(R.unit_dir("M1_ONE", out_root) / "checkpoints.pt", weights_only=True)
    model, _ = Mo.build_critic("MF_VCS_M1", *CASE, 4, 1017)
    fit = D.load_case_samples(record, suite, cfg.case_id, ("fit",))["fit"]
    A_, B_ = model.features(Mo.onehot(fit.x, 8), Mo.onehot(fit.y, 8))
    terms, info = A.mf_objective("MF_M1", [O.mf_u_scene(A_, B_)], model.W, [1.0], m1=A.M1Config(max_iterations=30))
    assert torch.equal(blob["states"][1]["W"], info["m1_W"])                                # the saved W is the solved head of the step
    log = C.read_json(R.unit_dir("M1_ONE", out_root) / "train_steps.json.gz")
    assert log["m1_iterations"][0] == info["m1_iterations"] and log["m1_gap_final"][0] == info["m1_gap_final"]


def test_training_is_deterministic(tiny_suite, out_root):
    suite, record = tiny_suite
    a = _run(_cfg("MLP_VCS_perm8", None, None, uid="DET_A"), suite, record, out_root)
    b = _run(_cfg("MLP_VCS_perm8", None, None, uid="DET_B"), suite, record, out_root)
    assert a["final_state_sha256"] == b["final_state_sha256"] and a["checkpoints"] == b["checkpoints"]
    c = _run(_cfg("MLP_VCS_perm8", None, None, uid="DET_C", algo=1018), suite, record, out_root)
    assert c["final_state_sha256"] != a["final_state_sha256"]


def test_unit_config_validation():
    with pytest.raises(ValueError):
        _cfg("MF_VCS_M1", 4, None)
    with pytest.raises(ValueError):
        _cfg("MF_VCS_M0", None, None)
    with pytest.raises(ValueError):
        _cfg("MLP_VCS_perm1", None, 10)
    with pytest.raises(ValueError):
        _cfg("MF_VCS_M0", 4, None, steps=5, ckpt=2)
