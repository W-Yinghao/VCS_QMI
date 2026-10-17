"""V10 T03 aggregator (analysis tool; v10/tools is outside the T03 code hash, as for the BF report tools).  Evaluator side: it reads
runs/<unit>/result.json and runs/<unit>/evaluator_only/evaluation.json; nothing here feeds back into any selection.  Every output that
carries a truth-derived value (S, population error, EVAL statistics) is written under aggregate/evaluator_only/; the cost record and the
budget carry none (statuses, check flags, costs, counts only).

  t03_aggregate.py smoke [--output-dir DIR]   smoke summary of one output root:
      aggregate/T03_SMOKE_COST.json                  truth-free: per-unit status, full cost (wall / CPU / per step / phases /
                                                     evaluation), implementation checks, the grid projection at this layout
      aggregate/evaluator_only/T03_SMOKE_SUMMARY.json  the same plus the evaluation values of every smoke unit
  t03_aggregate.py budget --layout NAME=ROOT:PROCESSES:THREADS [...] [--output PATH]
      truth-free grid budget from the smoke cost records of one or more process layouts of the same smoke units (the formal smoke =
      1 process x 8 threads; layout measurements = P concurrent processes x T threads on the same 8-CPU allocation): per-family cost,
      process-hours / CPU-hours of the 1212 units, wall time at the allowed concurrency, and the comparison with the phase-1 order of
      magnitude (V10_PHASE1_STAGE_REPORT_20261017.md section 5.1, quoted and checked against the report file)
  t03_aggregate.py grid  [--output PATH]      status coverage + the decision 9.3 / 9.4 / 9.5 tables (aggregate/evaluator_only/):
      main        per (condition, method) over the 10 data replicates (algo seed 1001): EVAL U-J - S bias, variance, RMSE with
                  finite-replicate uncertainty (t interval, chi-square interval, percentile bootstrap over replicates, seed 10191),
                  population E_M(T - eta)^2, J_pop - S, EVAL sampling error, head gap, permutation conditional variance; paired
                  method differences per data replicate (the primary pairing unit)
      algo        conditional algorithm variation on data root 2026101700: per (condition, method) over algo seeds 1001 (the main
                  unit of that data root) / 1002 / 1003 (algo_extra), R = 3; its own table, never pooled with the data replicates
      capacity    K 8 / 16 / 32 for the 4 capacity conditions x 3 MF methods over the 10 data replicates (algo 1001): K16 = the main
                  units of those conditions, K8 / K32 = the capacity units (fixed training rule; includes optimisation effects)
      saturated   the FIT-count saturated table per condition and data replicate (deduplicated across units; must agree)
      cost        per block / method
Incomplete cells are reported as such (n_complete < expected); no row is dropped or imputed.
"""
from __future__ import annotations

import argparse
import collections
import heapq
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "v10_t03"))
import t03_common as C  # noqa: E402
import t03_units as U   # noqa: E402

BOOT_B, BOOT_SEED = 2000, 10191                                  # package V10_DESIGN section 8 bootstrap constants
PAIRS = (("MF_VCS_M0", "MF_VCS_perm1"), ("MF_VCS_M1", "MF_VCS_M0"), ("MLP_VCS_perm8", "MLP_VCS_perm1"), ("MF_VCS_M0", "MLP_VCS_perm1"))
COST_KEYS = ("wall_seconds_unit_total", "cpu_seconds_unit_total", "wall_seconds_train_unit", "cpu_seconds_train_unit", "wall_seconds_loop",
             "wall_seconds_cal_fit_readout", "wall_seconds_checkpoint_copy", "wall_seconds_evaluation", "wall_seconds_sample_load",
             "wall_seconds_artifact_save", "peak_rss_kib_process", "torch_threads")
SMOKE_COST = "T03_SMOKE_COST.json"
SMOKE_SUMMARY = "T03_SMOKE_SUMMARY.json"
# phase-1 report section 5.1 (the "original order of magnitude", decision 9.4): quoted verbatim and checked against the report file
PHASE1_REPORT = REPO / "v10" / "V10_PHASE1_STAGE_REPORT_20261017.md"
PHASE1_QUOTES = ("非 M1 的 576 次训练约 0.64 作业小时", "M1 的 216 次训练每步另有 HVP 调用", "合计约 2.6/7.1/22.8 作业小时",
                 "1 个 CPU 作业（cpu-high、16 线程、--time=02:00:00")
PHASE1_ESTIMATE = {"non_M1": {"units": 576, "job_hours": 0.64}, "M1": {"units": 216, "job_hours_by_cap": {"10": 2.6, "30": 7.1, "100": 22.8}},
                   "job_threads": 16}
ALLOCATED_CPUS_SMOKE = 8                                          # slurm/v10_t03/t03_smoke.sbatch --cpus-per-task


def load_all(root: Path) -> dict[str, dict]:
    out = {}
    for row in U.load_rows():
        uid = row["unit_id"]
        d = root / "runs" / uid
        res = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else None
        ev_p = d / C.TRUTH_DIR_NAME / "evaluation.json"
        ev = json.loads(ev_p.read_text()) if ev_p.exists() else None
        out[uid] = {"row": row, "result": res, "evaluation": ev, "status": res["status"] if res else "NOT_RUN"}
    return out


def _stats(e, unit: str = "data replicate") -> dict:
    from scipy import stats
    e = np.asarray(e, dtype=np.float64)
    R = len(e)
    out = {"R": R}
    if R == 0:
        return out
    out.update({"mean": float(e.mean()), "rmse": float(np.sqrt(np.mean(e * e)))})
    if R >= 2:
        sd = float(e.std(ddof=1))
        tq = float(stats.t.ppf(0.975, R - 1))
        var = sd * sd
        boot = np.random.default_rng(BOOT_SEED).integers(0, R, (BOOT_B, R))
        rm = np.sqrt(np.mean(e[boot] ** 2, axis=1))
        out.update({"sd": sd, "se_mean": sd / math.sqrt(R), "mean_ci95_t": [out["mean"] - tq * sd / math.sqrt(R), out["mean"] + tq * sd / math.sqrt(R)],
                    "var": var, "var_ci95_chi2": [(R - 1) * var / float(stats.chi2.ppf(0.975, R - 1)), (R - 1) * var / float(stats.chi2.ppf(0.025, R - 1))],
                    "rmse_ci95_bootstrap": [float(np.percentile(rm, 2.5)), float(np.percentile(rm, 97.5))],
                    "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "unit": unit}})
    return out


def _sel(ev: dict) -> dict:
    return ev["selected_metrics"]


def _cell(units: list[dict], rep_key: str = "data_root_seed") -> dict:
    """One table cell over its replicates; rep_key = 'data_root_seed' (data replicates) or 'algo_seed' (conditional algorithm)."""
    label = {"data_root_seed": "data replicate", "algo_seed": "algorithm seed (fixed data)"}[rep_key]
    done = [u for u in units if u["status"] == "COMPLETE" and u["evaluation"] is not None]
    rows = sorted(((int(u["row"][rep_key]), _sel(u["evaluation"])) for u in done), key=lambda t: t[0])
    cell = {"expected": len(units), "n_complete": len(done), "status_counts": dict(collections.Counter(u["status"] for u in units)),
            "replicate_unit": label, "replicates": sorted(int(u["row"][rep_key]) for u in units), "replicates_complete": [s for s, _ in rows],
            "unit_ids": sorted(u["row"]["unit_id"] for u in units)}
    if rows:
        cell.update({"S": rows[0][1]["S"], "EVAL_UJ_minus_S": _stats([m["EVAL_UJ_minus_S"] for _, m in rows], label),
                     "pop_mse": _stats([m["pop_mse"] for _, m in rows], label),
                     "EVAL_UJ_minus_J_pop": _stats([m["EVAL_UJ_minus_J_pop"] for _, m in rows], label),
                     "EVAL_U_minus_V": _stats([m["EVAL_U_minus_V"] for _, m in rows], label),
                     "perm1_var_fit": _stats([m["perm_conditional_variance"]["fit"]["perm1_var"] for _, m in rows], label)})
        gaps = [m["head_gap"]["fw_gap_W"] for _, m in rows if m.get("head_gap", {}).get("status") == "MEASURED"]
        if gaps:
            cell["head_fw_gap_fit"] = _stats(gaps, label)
        cell["identity_all_pass"] = all(m["identity_check"] == "PASS" for _, m in rows)
    return cell


def _cond(r: dict) -> tuple:
    return (r["family"], int(r["m"]), float(r["rho"]))


def grid(root: Path) -> dict:
    allu = load_all(root)
    by_block = collections.defaultdict(list)
    for u in allu.values():
        by_block[u["row"]["block"]].append(u)
    out = {"schema": "v10_t03.aggregate.v2", "utc": C.utc_now(), "out_root": str(root),
           "status": {b: dict(collections.Counter(u["status"] for u in us)) for b, us in by_block.items()}}
    # main: (condition, method) over the 10 data replicates, algo 1001
    groups = collections.defaultdict(list)
    for u in by_block.get("main", []):
        groups[_cond(u["row"]) + (u["row"]["method"],)].append(u)
    out["main"] = [{"case_id": C.case_id(*k[:3]), "method": k[3], **_cell(us)} for k, us in sorted(groups.items())]
    # algo: (condition, method) on data root MAIN_DATA_SEEDS[0] over algo 1001 (main) / 1002 / 1003 (algo_extra)
    groups = collections.defaultdict(list)
    for u in by_block.get("main", []):
        if int(u["row"]["data_root_seed"]) == C.MAIN_DATA_SEEDS[0] and int(u["row"]["algo_seed"]) == C.MAIN_ALGO_SEED:
            groups[_cond(u["row"]) + (u["row"]["method"],)].append(u)
    for u in by_block.get("algo_extra", []):
        groups[_cond(u["row"]) + (u["row"]["method"],)].append(u)
    out["algo"] = [{"case_id": C.case_id(*k[:3]), "method": k[3], "data_root_seed": C.MAIN_DATA_SEEDS[0], **_cell(us, "algo_seed")}
                   for k, us in sorted(groups.items())]
    # capacity: (capacity condition, MF method, K) over the 10 data replicates; K16 = main units of the same condition / method
    groups = collections.defaultdict(list)
    for u in by_block.get("main", []):
        r = u["row"]
        if _cond(r) in C.CAPACITY_CONDITIONS and r["method"] in C.MF_METHODS:
            groups[_cond(r) + (r["method"], int(r["K"]))].append(u)
    for u in by_block.get("capacity", []):
        r = u["row"]
        groups[_cond(r) + (r["method"], int(r["K"]))].append(u)
    out["capacity"] = [{"case_id": C.case_id(*k[:3]), "method": k[3], "K": k[4], "source_block": "main" if k[4] == C.K_MAIN else "capacity",
                        **_cell(us)} for k, us in sorted(groups.items())]
    # paired method differences per data replicate (main)
    pairs = []
    mains = collections.defaultdict(dict)
    for u in by_block.get("main", []):
        if u["status"] == "COMPLETE" and u["evaluation"]:
            r = u["row"]
            mains[_cond(r)].setdefault(int(r["data_root_seed"]), {})[r["method"]] = _sel(u["evaluation"])
    for cond, reps in sorted(mains.items()):
        for a, b in PAIRS:
            common = sorted(s for s, mm in reps.items() if a in mm and b in mm)
            d_sq = [reps[s][a]["EVAL_UJ_minus_S"] ** 2 - reps[s][b]["EVAL_UJ_minus_S"] ** 2 for s in common]
            d_mse = [reps[s][a]["pop_mse"] - reps[s][b]["pop_mse"] for s in common]
            pairs.append({"case_id": C.case_id(*cond), "a": a, "b": b, "n_pairs": len(common), "diff_sq_err_EVAL_UJ": _stats(d_sq),
                          "diff_pop_mse": _stats(d_mse)})
    out["paired_main"] = pairs
    # saturated FIT table, deduplicated per (data root, condition)
    sat = {}
    conflicts = []
    for u in allu.values():
        if u["status"] != "COMPLETE" or not u["evaluation"]:
            continue
        r = u["row"]
        k = (int(r["data_root_seed"]),) + _cond(r)
        rec = {m: {f: u["evaluation"]["saturated_fit_table"][m][f] for f in ("pop_mse", "EVAL_UJ", "FIT_UJ", "CAL_UJ", "J_pop")} for m in ("U", "V")}
        if k in sat and sat[k] != rec:
            conflicts.append(list(k))
        sat.setdefault(k, rec)
    out["saturated"] = {"conflicts": conflicts, "rows": [{"data_root_seed": k[0], "case_id": C.case_id(*k[1:]), **v}
                                                         for k, v in sorted(sat.items())]}
    # cost
    cost = collections.defaultdict(list)
    for u in allu.values():
        if u["result"] and u["result"].get("cost", {}).get("wall_seconds_unit_total") is not None:
            cost[(u["row"]["block"], u["row"]["method"])].append(u["result"]["cost"]["wall_seconds_unit_total"])
    out["cost"] = [{"block": b, "method": m, "units": len(v), "wall_seconds_sum": float(np.sum(v)), "wall_seconds_mean": float(np.mean(v))}
                   for (b, m), v in sorted(cost.items())]
    return out


# ------------------------------------------------------------------------------------------------------------ smoke
def _cost_key(method: str, cap) -> str:
    return f"{method}|cap{cap}" if method == "MF_VCS_M1" else method


def smoke(root: Path) -> tuple[dict, dict]:
    """(truth-free cost record, evaluator-only summary) of the 7 smoke units of one output root."""
    allu = load_all(root)
    units = [u for u in allu.values() if u["row"]["block"] == "smoke"]
    cost_rows, eval_rows, checks = [], [], {}
    for u in units:
        r, res, ev = u["row"], u["result"], u["evaluation"]
        ent = {"unit_id": r["unit_id"], "method": r["method"], "cap": r["cap"], "K": r["K"], "status": u["status"], "steps_expected": int(r["steps"])}
        if res and res["status"] in ("COMPLETE", "FAIL") and "cost" in res and res.get("config"):
            c = res["cost"]
            ent.update({k: c.get(k) for k in COST_KEYS})
            ent.update({"step_seconds": c.get("step_seconds"), "phase_seconds": c.get("phase_seconds"),
                        "init_state_sha256": res.get("critic", {}).get("init_state_sha256"),
                        "perm_stream": (res.get("streams") or {}).get("perm"), "derangements": res.get("derangements"),
                        "selected_step_own_cal": (res.get("selection") or {}).get("own_cal", {}).get("step"),
                        "truth_barrier_denials": len((res.get("truth_barrier") or {}).get("denials_during_unit") or []),
                        "steps_completed": res.get("steps_completed")})
            if "m1" in res:
                ent["m1"] = res["m1"]
            if res["status"] == "FAIL":
                ent["failure"] = {k: v for k, v in res.get("failure", {}).items() if k != "traceback"}
        elif res and res["status"] in ("FAIL", "BLOCKED"):
            ent["failure"] = {k: v for k, v in (res.get("failure") or {"reason": res.get("reason")}).items() if k != "traceback"}
        if ev:
            ent["evaluation"] = {"status": ev["status"], "checks": ev["checks"], "seconds": ev.get("seconds")}
            sm = ev["selected_metrics"]
            eval_rows.append({"unit_id": r["unit_id"], "method": r["method"], "cap": r["cap"], "selected_step": ev["selected"]["step"],
                              "S": sm["S"], "pop_mse": sm["pop_mse"], "J_pop": sm["J_pop"], "EVAL_UJ": sm["EVAL_UJ"], "EVAL_VJ": sm["EVAL_VJ"],
                              "EVAL_UJ_minus_S": sm["EVAL_UJ_minus_S"], "identity_residual": sm["identity_residual"],
                              "head_gap": sm.get("head_gap"), "perm_conditional_variance": {k: {f: v[f] for f in ("perm1_var", "z_mean_vs_exact")}
                                                                                        for k, v in sm["perm_conditional_variance"].items()},
                              "saturated_U": {f: ev["saturated_fit_table"]["U"][f] for f in ("pop_mse", "EVAL_UJ", "FIT_UJ", "J_pop")},
                              "readout_reproduction": ev.get("readout_reproduction")})
        cost_rows.append(ent)
    ok = [e for e in cost_rows if e["status"] == "COMPLETE"]
    mf = {e["init_state_sha256"] for e in ok if e["method"].startswith("MF_")}
    mlp = {e["init_state_sha256"] for e in ok if e["method"].startswith("MLP_")}
    perm = {json.dumps(e["perm_stream"], sort_keys=True) for e in ok if e.get("perm_stream")}
    checks["all_smoke_units_complete"] = "PASS" if len(ok) == len(cost_rows) == 7 else "FAIL"
    checks["full_2000_steps"] = "PASS" if ok and all(e.get("steps_completed") == e["steps_expected"] == C.STEPS for e in ok) else "FAIL"
    checks["MF_methods_share_initial_state"] = "PASS" if len(mf) == 1 and sum(e["method"].startswith("MF_") for e in ok) == 5 else "FAIL"
    checks["MLP_perm1_perm8_share_initial_state"] = "PASS" if len(mlp) == 1 and sum(e["method"].startswith("MLP_") for e in ok) == 2 else "FAIL"
    checks["perm_methods_share_derangement_stream"] = "PASS" if len(perm) == 1 else "FAIL"
    checks["evaluations_pass"] = "PASS" if ok and all(e.get("evaluation", {}).get("status") == "PASS" for e in ok) else "FAIL"
    checks["no_truth_barrier_denials"] = "PASS" if ok and all(e.get("truth_barrier_denials") == 0 for e in cost_rows if "truth_barrier_denials" in e) else "FAIL"
    checks["M1_first_step_from_W0"] = "PASS" if all(e["m1"]["first_step_W_start_is_zero"] for e in ok if "m1" in e) else "FAIL"
    cost_of = {_cost_key(e["method"], e["cap"]): e["wall_seconds_unit_total"] for e in ok}
    threads = sorted({e["torch_threads"] for e in ok})
    env = next((u["result"]["provenance"]["environment"] for u in units if u["result"] and u["result"].get("provenance", {}).get("environment")), None)
    status = "PASS" if all(v == "PASS" for v in checks.values()) else "FAIL"
    cost = {"schema": "v10_t03.smoke_cost.v1", "utc": C.utc_now(), "out_root": str(root), "status": status, "checks": checks, "units": cost_rows,
            "per_unit_wall_seconds": cost_of, "torch_threads": threads,
            "grid_projection_this_layout": project(cost_of) if len(cost_of) == 7 else None,
            "environment": env, "note": "truth-free: statuses, check flags, costs and counts only; evaluation values are in "
                                        f"aggregate/{C.TRUTH_DIR_NAME}/{SMOKE_SUMMARY}"}
    summary = {"schema": "v10_t03.smoke_summary.v2", "utc": cost["utc"], "out_root": str(root), "status": status, "checks": checks,
               "evaluation": eval_rows, "cost_record": f"aggregate/{SMOKE_COST}"}
    return cost, summary


# ------------------------------------------------------------------------------------------------------------ projection / budget
def grid_rows() -> list[dict]:
    return [r for r in U.load_rows() if r["block"] != "smoke"]


def unit_cost(row: dict, cost_of: dict, bound: str) -> float:
    """Projected seconds of one grid row from the smoke unit with the same method (and cap); formal M1 rows (cap CAL / CAL_K16) take
    the min ('low') or max ('high') over the three caps; m = 8 and K = 8 / 32 rows are charged the m = 32, K = 16 smoke cost."""
    meth, cap = row["method"], row["cap"]
    if meth == "MF_VCS_M1" and cap in ("CAL", "CAL_K16"):
        vals = [cost_of[_cost_key(meth, str(c))] for c in C.M1_CAPS]
        return min(vals) if bound == "low" else max(vals)
    return cost_of[_cost_key(meth, cap if meth == "MF_VCS_M1" else "NA")]


def project(cost_of: dict) -> dict:
    """Sequential process-hours of the 1212 grid units at the measured per-unit cost of one layout (low / high = the M1 cap bound)."""
    rows = grid_rows()
    out = {"units": dict(collections.Counter(r["block"] for r in rows)), "n_units": len(rows),
           "process_hours": {}, "by_block_process_hours": {}, "by_method_process_hours": {},
           "basis": "per-unit wall seconds of the smoke unit with the same method (and cap); formal M1 rows use the min (low) / max (high) "
                    "over caps because their cap is CAL-selected; m = 8 and K = 8 / 32 rows are charged the m = 32, K = 16 smoke cost "
                    "(proxy, not measured); queue time excluded", "status": "PROJECTION_NOT_MEASUREMENT"}
    for bound in ("low", "high"):
        sec = [(r, unit_cost(r, cost_of, bound)) for r in rows]
        out["process_hours"][bound] = sum(s for _, s in sec) / 3600
        blk, meth = collections.Counter(), collections.Counter()
        for r, s in sec:
            blk[r["block"]] += s / 3600
            meth[r["method"]] += s / 3600
        out["by_block_process_hours"][bound] = dict(blk)
        out["by_method_process_hours"][bound] = dict(meth)
    return out


def lpt_makespan(durations: list[float], workers: int) -> float:
    """Longest-processing-time-first greedy schedule of independent tasks on identical workers (seconds)."""
    if not durations:
        return 0.0
    heap = [0.0] * max(1, int(workers))
    for d in sorted(durations, reverse=True):
        heapq.heappush(heap, heapq.heappop(heap) + d)
    return max(heap)


def phase1_order_of_magnitude() -> dict:
    """The phase-1 report section 5.1 estimate (head-only proxy), its quotes checked against the report file, scaled per unit to the
    current grid's M1 / non-M1 counts."""
    text = PHASE1_REPORT.read_text(encoding="utf-8")
    missing = [q for q in PHASE1_QUOTES if q not in text]
    rows = grid_rows()
    n_m1 = sum(r["method"] == "MF_VCS_M1" for r in rows)
    n_other = len(rows) - n_m1
    e = PHASE1_ESTIMATE
    per_other = e["non_M1"]["job_hours"] / e["non_M1"]["units"]
    scaled = {cap: n_other * per_other + n_m1 * h / e["M1"]["units"] for cap, h in e["M1"]["job_hours_by_cap"].items()}
    return {"source": str(PHASE1_REPORT.relative_to(REPO)) + " section 5.1", "quotes": list(PHASE1_QUOTES),
            "quotes_found_in_report": not missing, "missing_quotes": missing, "estimate": e,
            "estimate_original_total_job_hours_by_cap": {cap: e["non_M1"]["job_hours"] + h for cap, h in e["M1"]["job_hours_by_cap"].items()},
            "scaled_to_current_grid": {"units": {"non_M1": n_other, "M1": n_m1}, "job_hours_by_cap": scaled,
                                       "cpu_hours_by_cap": {cap: h * e["job_threads"] for cap, h in scaled.items()},
                                       "rule": "per-unit job-hours of the phase-1 estimate x the current unit counts; job = 16 threads"},
            "nature": "rough head-only estimate of the phase-1 report (not measured; '只给数量级')"}


def phase1_comparison(cpu_hours: dict, om: dict) -> dict:
    """Directional comparison with the phase-1 order of magnitude (scaled to the current grid, CPU-hours).  Decision 9.4 / protocol P2
    t03.smoke.over_budget act only when the measured resources EXCEED the original order of magnitude (> 10 x its dearest scenario)."""
    ref = om["scaled_to_current_grid"]["cpu_hours_by_cap"]
    lo, hi = cpu_hours["low"], cpu_hours["high"]
    exceeds = bool(hi > 10 * ref["100"])
    return {"cpu_hours_low_over_scaled_cap10": lo / ref["10"], "cpu_hours_high_over_scaled_cap100": hi / ref["100"],
            "exceeds_phase1_order_of_magnitude": exceeds, "below_phase1_order_of_magnitude": bool(lo < ref["10"] / 10),
            "within_phase1_scaled_band": bool(ref["10"] <= lo and hi <= ref["100"]),
            "over_budget_rule_triggered": exceeds,
            "rule": "exceeds = projected CPU-h (high) > 10 x the scaled cap-100 scenario; below = projected CPU-h (low) < the scaled cap-10 "
                    "scenario / 10; only 'exceeds' triggers the over-budget rule (record the new budget and the completable scope first)"}


def first_steps_excess(root: Path, procs: int) -> dict:
    """Per smoke unit of one layout: seconds of the first 100 steps above the median step of steps 201-2000 (process warm-up or the
    early head-solver iterations of M1), and whether the unit was the first unit of its process (shard order k::P).  Descriptive only:
    the projections use the measured walls unchanged."""
    import gzip
    out = {}
    ids = [r["unit_id"] for r in U.load_rows() if r["block"] == "smoke"]
    for i, uid in enumerate(ids):
        p = root / "runs" / uid / "train_steps.json.gz"
        if not p.exists():
            out[uid] = {"status": "NOT_AVAILABLE"}
            continue
        t = np.asarray(json.loads(gzip.decompress(p.read_bytes()))["t_step"], dtype=np.float64)
        steady = float(np.median(t[200:])) if len(t) > 200 else float(np.median(t))
        out[uid] = {"first_unit_of_process": bool(i < procs), "steady_step_seconds": steady,
                    "excess_seconds_first_100_steps": float(np.clip(t[:100] - steady, 0, None).sum())}
    return out


def budget(layouts: list[tuple[str, Path, int, int]], jobs=(1, 2), cpus_per_job=(32, 64)) -> dict:
    """Truth-free grid budget from the smoke cost records of the given layouts (name, root, processes, threads)."""
    rows = grid_rows()
    om = phase1_order_of_magnitude()
    out = {"schema": "v10_t03.budget.v1", "utc": C.utc_now(), "status": "MEASURED_PROJECTION", "layouts": [], "phase1_order_of_magnitude": om,
           "allowed_concurrency": {"rule": "at most 2 queued V10 jobs (job names v10_*, BF and T03 together); normal-QOS count <= 29",
                                   "cpu_high": "2 nodes x 96 CPUs (48 cores x 2 threads)"},
           "assumptions": ["grid units charged the per-unit wall of the smoke unit with the same method (and cap) measured in the same "
                           "process layout; m = 8 and K = 8 / 32 rows charged the m = 32 / K = 16 cost (proxy)",
                           "formal M1 rows (CAL-selected cap) bounded by the cheapest (low) and the dearest (high) cap",
                           "CPU-hours = process-hours x allocated CPUs per process (allocation / processes)",
                           "wall time: two phases (252 CAL units, then the 960 formal units, which need the CAL selection), each an LPT "
                           "schedule over jobs x processes; per-unit cost assumed unchanged at the larger process count of a bigger "
                           "allocation (memory-bandwidth contention beyond the measured 8-CPU layout is not measured); queue time excluded",
                           "measured walls are used unchanged: a one-off process warm-up inside a measured unit (first_steps_excess_seconds) "
                           "is charged to every projected unit of that family (conservative)"]}
    cal = [r for r in rows if r["block"] == "CAL"]
    formal = [r for r in rows if r["block"] != "CAL"]
    for name, root, procs, threads in layouts:
        p = root / "aggregate" / SMOKE_COST
        if not p.exists():
            out["layouts"].append({"name": name, "status": "NOT_RUN", "cost_record": str(p)})
            continue
        rec = json.loads(p.read_text())
        ent = {"name": name, "cost_record": str(p), "cost_record_sha256": C.sha256_file(p), "processes": procs, "threads_per_process": threads,
               "allocated_cpus": ALLOCATED_CPUS_SMOKE, "cpus_per_process": ALLOCATED_CPUS_SMOKE / procs, "smoke_status": rec["status"],
               "smoke_checks": rec["checks"], "torch_threads_recorded": rec["torch_threads"]}
        cost_of = rec["per_unit_wall_seconds"]
        if rec["status"] != "PASS" or len(cost_of) != 7:
            ent["status"] = "INCOMPLETE"
            out["layouts"].append(ent)
            continue
        cpp = ent["cpus_per_process"]
        by_key = {_cost_key(e["method"], e["cap"]): e for e in rec["units"] if e["status"] == "COMPLETE"}
        ent["per_family"] = {k: {"unit_id": by_key[k]["unit_id"], "wall_seconds_unit_total": v, "cpu_hours_allocated_per_unit": v * cpp / 3600,
                                 **{f: by_key[k].get(f) for f in ("wall_seconds_train_unit", "wall_seconds_evaluation", "cpu_seconds_unit_total")}}
                             for k, v in sorted(cost_of.items())}
        proj = rec["grid_projection_this_layout"]
        ent["process_hours"] = proj["process_hours"]
        ent["cpu_hours"] = {b: h * cpp for b, h in proj["process_hours"].items()}
        ent["by_block_process_hours"] = proj["by_block_process_hours"]
        ent["by_method_process_hours"] = proj["by_method_process_hours"]
        walls = []
        for J in jobs:
            for cpus in cpus_per_job:
                workers = J * int(cpus // cpp)
                w = {b: (lpt_makespan([unit_cost(r, cost_of, b) for r in cal], workers)
                         + lpt_makespan([unit_cost(r, cost_of, b) for r in formal], workers)) / 3600 for b in ("low", "high")}
                walls.append({"jobs": J, "cpus_per_job": cpus, "processes_total": workers, "wall_hours": w})
        ent["wall_hours_at_concurrency"] = walls
        ent["phase1_comparison"] = phase1_comparison(ent["cpu_hours"], om)
        ent["first_steps_excess_seconds"] = first_steps_excess(root, procs)
        ent["status"] = "MEASURED_PROJECTION"
        out["layouts"].append(ent)
    return out


# ------------------------------------------------------------------------------------------------------------ CLI
def _layout(spec: str) -> tuple[str, Path, int, int]:
    m = re.fullmatch(r"([A-Za-z0-9_]+)=(.+):(\d+):(\d+)", spec)
    if not m:
        raise SystemExit(f"bad --layout {spec!r} (NAME=ROOT:PROCESSES:THREADS)")
    return m.group(1), Path(m.group(2)), int(m.group(3)), int(m.group(4))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("smoke", "grid", "budget"))
    ap.add_argument("--output", type=Path, help="grid / budget output path")
    ap.add_argument("--output-dir", type=Path, help="smoke: the aggregate directory (default <out root>/aggregate)")
    ap.add_argument("--layout", action="append", default=[], help="budget: NAME=ROOT:PROCESSES:THREADS (repeatable)")
    a = ap.parse_args(argv)
    root = C.out_root()
    agg = a.output_dir or (root / "aggregate")
    tool_sha = C.sha256_file(Path(__file__))
    if a.what == "smoke":
        cost, summary = smoke(root)
        cost["aggregator_sha256"] = summary["aggregator_sha256"] = tool_sha
        C.write_json_atomic(agg / SMOKE_COST, cost)
        C.write_json_atomic(agg / C.TRUTH_DIR_NAME / SMOKE_SUMMARY, summary)
        print(json.dumps({"cost_record": str(agg / SMOKE_COST), "summary": str(agg / C.TRUTH_DIR_NAME / SMOKE_SUMMARY), "status": cost["status"],
                          "checks": cost["checks"], "per_unit_wall_seconds": cost["per_unit_wall_seconds"],
                          "process_hours": (cost["grid_projection_this_layout"] or {}).get("process_hours")}, indent=1))
        return 0 if cost["status"] == "PASS" else 1
    if a.what == "budget":
        if not a.layout:
            raise SystemExit("budget needs at least one --layout")
        res = budget([_layout(s) for s in a.layout])
        res["aggregator_sha256"] = tool_sha
        path = a.output or (agg / "T03_BUDGET.json")
        C.write_json_atomic(path, res)
        print(json.dumps({"output": str(path), "layouts": [{**{k: L.get(k) for k in ("name", "status", "process_hours", "cpu_hours")},
                                                            "over_budget_rule_triggered": (L.get("phase1_comparison") or {}).get("over_budget_rule_triggered")}
                                                           for L in res["layouts"]],
                          "phase1_quotes_found": res["phase1_order_of_magnitude"]["quotes_found_in_report"]}, indent=1))
        return 0 if all(L["status"] == "MEASURED_PROJECTION" for L in res["layouts"]) and res["phase1_order_of_magnitude"]["quotes_found_in_report"] else 1
    res = grid(root)
    res["aggregator_sha256"] = tool_sha
    path = a.output or (agg / C.TRUTH_DIR_NAME / "T03_AGGREGATE.json")
    C.write_json_atomic(path, res)
    print(json.dumps({"output": str(path), "block_status": res["status"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
