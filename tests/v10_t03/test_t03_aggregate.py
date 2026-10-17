"""The aggregator (v10/tools/t03_aggregate.py) on tiny stand-ins for the 7 smoke units (4 steps each): smoke checks, the truth-free cost
record vs the evaluator-only summary, no truth value outside evaluator-only paths, the grid tables (decision 9.3 / 9.4 grouping) and
the budget arithmetic."""
from __future__ import annotations

import gzip
import importlib.util
import json

import pytest

import t03_common as C
import t03_run as R
import t03_train as TR
import t03_units as U

spec = importlib.util.spec_from_file_location("t03_aggregate", C.REPO / "v10" / "tools" / "t03_aggregate.py")
AG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AG)


@pytest.fixture()
def smoke_standins(tiny_suite, out_root):
    suite, record = tiny_suite
    ident = C.identity()
    for row in U.load_rows():
        if row["block"] != "smoke":
            continue
        r = U.parse(row)
        cfg = TR.UnitConfig(unit_id=r["unit_id"], block="smoke", method=r["method"], family=r["family"], m=r["m"], rho=r["rho"], K=r["K"],
                            lr=r["lr"], cap=r["cap"], data_root_seed=7, algo_seed=r["algo_seed"], steps=4, ckpt_every=2)
        res = R.execute_unit(cfg, suite=suite, record=record, out=R.unit_dir(cfg.unit_id), ident=ident, env={"note": "test"}, extra={"unit_row": row})
        assert res["status"] == "COMPLETE", res.get("failure")
    return out_root


def test_smoke_cost_record_and_summary(smoke_standins):
    cost, summary = AG.smoke(smoke_standins)
    failing = {k for k, v in cost["checks"].items() if v != "PASS"}
    assert failing == {"full_2000_steps"} and cost["status"] == "FAIL"      # 4-step stand-ins are not full trainings: caught
    assert len(cost["units"]) == 7 and all(u["wall_seconds_unit_total"] > 0 and u["truth_barrier_denials"] == 0 for u in cost["units"])
    assert set(cost["per_unit_wall_seconds"]) == {"MF_VCS_perm1", "MF_VCS_M0", "MF_VCS_M1|cap10", "MF_VCS_M1|cap30", "MF_VCS_M1|cap100",
                                                   "MLP_VCS_perm1", "MLP_VCS_perm8"}
    p = cost["grid_projection_this_layout"]
    assert p["status"] == "PROJECTION_NOT_MEASUREMENT" and p["units"] == {"CAL": 252, "main": 600, "algo_extra": 120, "capacity": 240}
    assert 0 < p["process_hours"]["low"] <= p["process_hours"]["high"]
    assert len(summary["evaluation"]) == 7 and all("S" in e and "pop_mse" in e for e in summary["evaluation"])
    for u in cost["units"]:                                                  # the cost record carries no evaluation value
        assert set(u.get("evaluation", {})) <= {"status", "checks", "seconds"}
    json.dumps(C.jsonable(cost), allow_nan=False)


def _walk_text(root):
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix in (".npz", ".pt"):
            continue
        raw = p.read_bytes()
        if p.suffix == ".gz":
            raw = gzip.decompress(raw)
        yield p, raw.decode("utf-8")


def test_no_truth_value_outside_evaluator_only(smoke_standins, tiny_suite):
    """Every file of the output root outside an evaluator_only path is free of the suite's S values and of the EVAL / population
    values of every unit's evaluation (positive control: they are found in the evaluator-only files)."""
    suite, record = tiny_suite
    root = smoke_standins
    R.ensure_checked_suite(7, root)                                           # checker summary + evaluator-only full record
    assert AG.main(["smoke"]) == 1                                            # FAIL only for the 4-step stand-ins (test above)
    assert AG.main(["grid"]) == 0
    needles = set()
    for cid in record["cases"]:
        S = json.loads((suite / cid / "evaluator_only" / "truth.json").read_text())["S"]
        if S != 0:
            needles.add(repr(float(S)))
    for ev_path in root.glob("runs/*/evaluator_only/evaluation.json"):
        sm = json.loads(ev_path.read_text())["selected_metrics"]
        needles |= {repr(float(sm[k])) for k in ("pop_mse", "J_pop", "J_pop_minus_S", "EVAL_UJ", "EVAL_VJ", "EVAL_UJ_minus_S") if sm[k] != 0}
    assert len(needles) > 20
    found_inside, leaks = set(), []
    for p, text in _walk_text(root):
        hits = {n for n in needles if n in text}
        if C.TRUTH_DIR_NAME in p.relative_to(root).parts:
            found_inside |= hits
        elif hits:
            leaks.append((str(p.relative_to(root)), sorted(hits)[:3]))
    assert not leaks, leaks
    assert found_inside == needles                                            # the scan does see them where they belong
    assert (root / "aggregate" / AG.SMOKE_COST).exists() and (root / "aggregate" / C.TRUTH_DIR_NAME / AG.SMOKE_SUMMARY).exists()
    assert (root / "aggregate" / C.TRUTH_DIR_NAME / "T03_AGGREGATE.json").exists()


def test_grid_tables_report_missing_units(smoke_standins):
    g = AG.grid(smoke_standins)
    assert g["status"]["smoke"] == {"COMPLETE": 7} and g["status"]["main"] == {"NOT_RUN": 600}
    assert len(g["main"]) == 60 and all(c["n_complete"] == 0 and c["expected"] == 10 for c in g["main"])
    assert len(g["algo"]) == 60 and all(c["expected"] == 3 and c["replicates"] == [1001, 1002, 1003] for c in g["algo"])
    assert len(g["capacity"]) == 36 and all(c["expected"] == 10 for c in g["capacity"])
    assert sorted({c["K"] for c in g["capacity"]}) == [8, 16, 32] and {c["source_block"] for c in g["capacity"] if c["K"] == 16} == {"main"}
    assert g["saturated"]["conflicts"] == [] and len(g["saturated"]["rows"]) == 1


def _fake_eval(v: float, data_seed: int) -> dict:
    sat = {"pop_mse": 1e-3 * (data_seed % 100 + 1), "EVAL_UJ": 0.1, "FIT_UJ": 0.2, "CAL_UJ": 0.15, "J_pop": 0.12}
    return {"selected_metrics": {"S": 0.5, "EVAL_UJ_minus_S": v, "pop_mse": abs(v), "EVAL_UJ_minus_J_pop": v / 2, "EVAL_U_minus_V": v / 4,
                                 "perm_conditional_variance": {"fit": {"perm1_var": v * v}}, "head_gap": {"status": "MEASURED", "fw_gap_W": 1e-6},
                                 "identity_check": "PASS"}, "saturated_fit_table": {"U": sat, "V": sat}}


def test_grid_grouping_follows_decision_9_3_and_9_4(out_root):
    """Fabricated COMPLETE records: the algo cell pools algo 1001 (main, data 2026101700) with algo_extra 1002 / 1003 (R = 3); the
    capacity table takes K16 from the main units of the capacity conditions; data replicates and algorithm seeds are never pooled."""
    rows = [U.parse(r) for r in U.load_rows()]
    cond, meth = ("two_block", 32, 0.25), "MF_VCS_M0"
    pick = lambda blk, **kw: [r for r in rows if r["block"] == blk and (r["family"], r["m"], r["rho"]) == cond and r["method"] == meth
                              and all(r[k] == v for k, v in kw.items())]
    chosen = (pick("main", data_root_seed=2026101700) + pick("main", data_root_seed=2026101701) + pick("algo_extra")
              + pick("capacity", K=8, data_root_seed=2026101700))
    assert len(chosen) == 5
    for i, r in enumerate(chosen):
        d = out_root / "runs" / r["unit_id"]
        C.write_json_atomic(d / "result.json", {"status": "COMPLETE", "cost": {"wall_seconds_unit_total": 1.0}})
        C.write_json_atomic(d / "evaluator_only" / "evaluation.json", _fake_eval(0.01 * (i + 1), r["data_root_seed"]))
    g = AG.grid(out_root)
    cid = C.case_id(*cond)
    main = next(c for c in g["main"] if c["case_id"] == cid and c["method"] == meth)
    assert main["n_complete"] == 2 and main["replicates_complete"] == [2026101700, 2026101701] and main["EVAL_UJ_minus_S"]["R"] == 2
    algo = next(c for c in g["algo"] if c["case_id"] == cid and c["method"] == meth)
    assert algo["n_complete"] == 3 and algo["replicates_complete"] == [1001, 1002, 1003] and algo["EVAL_UJ_minus_S"]["R"] == 3
    assert algo["EVAL_UJ_minus_S"]["bootstrap"]["unit"] == "algorithm seed (fixed data)"
    cap = {c["K"]: c for c in g["capacity"] if c["case_id"] == cid and c["method"] == meth}
    assert sorted(cap) == [8, 16, 32] and cap[16]["n_complete"] == 2 and cap[8]["n_complete"] == 1 and cap[32]["n_complete"] == 0
    assert g["saturated"]["conflicts"] == []


def _fake_cost_record(root, cost_of: dict, status="PASS"):
    units = [{"unit_id": f"U{i}", "method": k.split("|")[0], "cap": k.split("cap")[1] if "|" in k else "NA", "status": "COMPLETE",
              "wall_seconds_unit_total": v, "wall_seconds_train_unit": v * 0.9, "wall_seconds_evaluation": v * 0.1, "cpu_seconds_unit_total": v}
             for i, (k, v) in enumerate(sorted(cost_of.items()))]
    C.write_json_atomic(root / "aggregate" / AG.SMOKE_COST, {"status": status, "checks": {"x": status}, "units": units, "per_unit_wall_seconds": cost_of,
                                                            "torch_threads": [2], "grid_projection_this_layout": AG.project(cost_of)})


def test_budget_arithmetic(tmp_path):
    cost_of = {"MF_VCS_perm1": 1.0, "MF_VCS_M0": 1.0, "MLP_VCS_perm1": 1.0, "MLP_VCS_perm8": 1.0,
               "MF_VCS_M1|cap10": 2.0, "MF_VCS_M1|cap30": 3.0, "MF_VCS_M1|cap100": 4.0}
    _fake_cost_record(tmp_path / "a", cost_of)
    b = AG.budget([("p4t2", tmp_path / "a", 4, 2), ("gone", tmp_path / "missing", 8, 1)])
    L, gone = b["layouts"]
    assert gone["status"] == "NOT_RUN" and L["status"] == "MEASURED_PROJECTION" and L["cpus_per_process"] == 2
    # 880 non-M1 units x 1 s; 108 CAL M1 units = 36 per cap; 224 formal M1 units at the cheapest (2 s) / dearest (4 s) cap
    assert L["process_hours"]["low"] * 3600 == pytest.approx(880 + 36 * (2 + 3 + 4) + 224 * 2)
    assert L["process_hours"]["high"] * 3600 == pytest.approx(880 + 36 * (2 + 3 + 4) + 224 * 4)
    assert L["cpu_hours"]["high"] == pytest.approx(2 * L["process_hours"]["high"])
    for w in L["wall_hours_at_concurrency"]:
        n = w["processes_total"]
        assert n == w["jobs"] * w["cpus_per_job"] // 2
        for bd in ("low", "high"):
            assert L["process_hours"][bd] / n <= w["wall_hours"][bd] + 1e-12 and w["wall_hours"][bd] <= L["process_hours"][bd]
    om = b["phase1_order_of_magnitude"]
    assert om["quotes_found_in_report"] and om["scaled_to_current_grid"]["units"] == {"non_M1": 880, "M1": 332}
    pc = L["phase1_comparison"]                       # ~1 CPU-h against 80 - 576 scaled: far below, never 'exceeds'
    assert pc["below_phase1_order_of_magnitude"] and not pc["exceeds_phase1_order_of_magnitude"] and not pc["over_budget_rule_triggered"]
    ref = om["scaled_to_current_grid"]["cpu_hours_by_cap"]
    big = AG.phase1_comparison({"low": 11 * ref["100"], "high": 11 * ref["100"]}, om)
    assert big["exceeds_phase1_order_of_magnitude"] and big["over_budget_rule_triggered"] and not big["below_phase1_order_of_magnitude"]
    assert AG.phase1_comparison({"low": ref["10"], "high": ref["100"]}, om)["within_phase1_scaled_band"]
    assert all(v == {"status": "NOT_AVAILABLE"} for v in L["first_steps_excess_seconds"].values())          # no step logs here
    assert AG.lpt_makespan([5, 4, 3, 3, 3], 2) == 10 and AG.lpt_makespan([], 3) == 0


def test_stats_helper():
    st = AG._stats([0.1, -0.1, 0.2, 0.0])
    assert st["R"] == 4 and st["mean"] == pytest.approx(0.05) and st["rmse"] == pytest.approx((0.06 / 4) ** 0.5)
    assert st["mean_ci95_t"][0] < st["mean"] < st["mean_ci95_t"][1] and st["var_ci95_chi2"][0] < st["var"] < st["var_ci95_chi2"][1]
