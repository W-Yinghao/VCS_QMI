"""V10 T03 aggregator (analysis tool; v10/tools is outside the T03 code hash, as for the BF report tools).  Evaluator side: it reads
runs/<unit>/result.json and runs/<unit>/evaluator_only/evaluation.json; nothing here feeds back into any selection.

  t03_aggregate.py smoke [--output PATH]   smoke summary: per-unit status, full training cost (wall / CPU / per step / phases /
                                           evaluation), implementation checks, and a cost projection of the 1212-unit grid
  t03_aggregate.py grid  [--output PATH]   status coverage + the decision 9.3 / 9.4 / 9.5 tables (skeleton):
      main        per (condition, method) over the 10 data replicates (algo seed 1001): EVAL U-J - S bias, variance, RMSE with
                  finite-replicate uncertainty (t interval, chi-square interval, percentile bootstrap over replicates, seed 10191),
                  population E_M(T - eta)^2, J_pop - S, EVAL sampling error, head gap, permutation conditional variance; paired
                  method differences per data replicate (the primary pairing unit)
      algo_extra  conditional algorithm variation on data root 2026101700 (algo 1001 / 1002 / 1003): own table, never pooled
      capacity    K 8 / 16 / 32 for the 4 capacity conditions x 3 MF methods (fixed training rule; includes optimisation effects)
      saturated   the FIT-count saturated table per condition and data replicate (deduplicated across units; must agree)
      cost        per block / method
Incomplete cells are reported as such (n_complete < expected); no row is dropped or imputed.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "v10_t03"))
import t03_common as C  # noqa: E402
import t03_units as U   # noqa: E402

BOOT_B, BOOT_SEED = 2000, 10191                                  # package V10_DESIGN section 8 bootstrap constants
PAIRS = (("MF_VCS_M0", "MF_VCS_perm1"), ("MF_VCS_M1", "MF_VCS_M0"), ("MLP_VCS_perm8", "MLP_VCS_perm1"), ("MF_VCS_M0", "MLP_VCS_perm1"))


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


def _stats(e: np.ndarray) -> dict:
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
                    "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "data replicate"}})
    return out


def _sel(ev: dict) -> dict:
    return ev["selected_metrics"]


def _cell(units: list[dict]) -> dict:
    done = [u for u in units if u["status"] == "COMPLETE" and u["evaluation"] is not None]
    rows = [(int(u["row"]["data_root_seed"]), _sel(u["evaluation"])) for u in done]
    rows.sort()
    cell = {"expected": len(units), "n_complete": len(done), "status_counts": dict(collections.Counter(u["status"] for u in units)),
            "data_root_seeds": [s for s, _ in rows]}
    if rows:
        S = rows[0][1]["S"]
        cell.update({"S": S, "EVAL_UJ_minus_S": _stats([m["EVAL_UJ_minus_S"] for _, m in rows]),
                     "pop_mse": _stats([m["pop_mse"] for _, m in rows]), "EVAL_UJ_minus_J_pop": _stats([m["EVAL_UJ_minus_J_pop"] for _, m in rows]),
                     "EVAL_U_minus_V": _stats([m["EVAL_U_minus_V"] for _, m in rows]),
                     "perm1_var_fit": _stats([m["perm_conditional_variance"]["fit"]["perm1_var"] for _, m in rows])})
        gaps = [m["head_gap"]["fw_gap_W"] for _, m in rows if m.get("head_gap", {}).get("status") == "MEASURED"]
        if gaps:
            cell["head_fw_gap_fit"] = _stats(gaps)
        cell["identity_all_pass"] = all(m["identity_check"] == "PASS" for _, m in rows)
    return cell


def grid(root: Path) -> dict:
    allu = load_all(root)
    by_block = collections.defaultdict(list)
    for u in allu.values():
        by_block[u["row"]["block"]].append(u)
    out = {"schema": "v10_t03.aggregate.v1", "utc": C.utc_now(), "out_root": str(root),
           "status": {b: dict(collections.Counter(u["status"] for u in us)) for b, us in by_block.items()}}

    def key(u):
        r = u["row"]
        return (r["family"], int(r["m"]), float(r["rho"]), r["method"])

    for block in ("main", "algo_extra", "capacity"):
        groups = collections.defaultdict(list)
        for u in by_block.get(block, []):
            k = key(u) + ((int(u["row"]["K"]),) if block == "capacity" else ()) + ((int(u["row"]["algo_seed"]),) if block == "algo_extra" else ())
            groups[k].append(u)
        out[block] = [{"case_id": C.case_id(k[0], k[1], k[2]), "method": k[3], **({"K": k[4]} if block == "capacity" else {}),
                       **({"algo_seed": k[4]} if block == "algo_extra" else {}), **_cell(us)} for k, us in sorted(groups.items())]
    # paired method differences per data replicate (main)
    pairs = []
    mains = collections.defaultdict(dict)
    for u in by_block.get("main", []):
        if u["status"] == "COMPLETE" and u["evaluation"]:
            r = u["row"]
            mains[(r["family"], int(r["m"]), float(r["rho"]))].setdefault(int(r["data_root_seed"]), {})[r["method"]] = _sel(u["evaluation"])
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
        k = (int(r["data_root_seed"]), r["family"], int(r["m"]), float(r["rho"]))
        rec = {m: {f: u["evaluation"]["saturated_fit_table"][m][f] for f in ("pop_mse", "EVAL_UJ", "FIT_UJ", "CAL_UJ", "J_pop")} for m in ("U", "V")}
        if k in sat and sat[k] != rec:
            conflicts.append(list(k))
        sat.setdefault(k, rec)
    out["saturated"] = {"conflicts": conflicts, "rows": [{"data_root_seed": k[0], "case_id": C.case_id(k[1], k[2], k[3]), **v}
                                                         for k, v in sorted(sat.items())]}
    # cost
    cost = collections.defaultdict(list)
    for u in allu.values():
        if u["result"] and u["result"].get("cost", {}).get("wall_seconds_unit_total") is not None:
            cost[(u["row"]["block"], u["row"]["method"])].append(u["result"]["cost"]["wall_seconds_unit_total"])
    out["cost"] = [{"block": b, "method": m, "units": len(v), "wall_seconds_sum": float(np.sum(v)), "wall_seconds_mean": float(np.mean(v))}
                   for (b, m), v in sorted(cost.items())]
    return out


def smoke(root: Path) -> dict:
    allu = load_all(root)
    units = [u for u in allu.values() if u["row"]["block"] == "smoke"]
    rows, checks = [], {}
    for u in units:
        r, res, ev = u["row"], u["result"], u["evaluation"]
        ent = {"unit_id": r["unit_id"], "method": r["method"], "cap": r["cap"], "status": u["status"]}
        if res and res["status"] in ("COMPLETE", "FAIL") and "cost" in res and res.get("config"):
            c = res["cost"]
            ent.update({k: c.get(k) for k in ("wall_seconds_unit_total", "cpu_seconds_unit_total", "wall_seconds_train_unit", "wall_seconds_loop",
                                               "wall_seconds_cal_fit_readout", "wall_seconds_checkpoint_copy", "wall_seconds_evaluation",
                                               "wall_seconds_sample_load", "wall_seconds_artifact_save", "peak_rss_kib_process", "torch_threads")})
            ent["step_seconds"] = c.get("step_seconds")
            ent["phase_seconds"] = c.get("phase_seconds")
            ent["init_state_sha256"] = res.get("critic", {}).get("init_state_sha256")
            ent["perm_stream"] = (res.get("streams") or {}).get("perm")
            ent["derangements"] = res.get("derangements")
            if "m1" in res:
                ent["m1"] = res["m1"]
            if res["status"] == "FAIL":
                ent["failure"] = {k: v for k, v in res.get("failure", {}).items() if k != "traceback"}
        if ev:
            sm = ev["selected_metrics"]
            ent["evaluation"] = {"status": ev["status"], "checks": ev["checks"], "selected_step": ev["selected"]["step"], "S": sm["S"],
                                 "pop_mse": sm["pop_mse"], "EVAL_UJ": sm["EVAL_UJ"], "EVAL_VJ": sm["EVAL_VJ"], "identity_residual": sm["identity_residual"],
                                 "readout_reproduction": ev.get("readout_reproduction"), "seconds": ev.get("seconds")}
        rows.append(ent)
    ok = [e for e in rows if e["status"] == "COMPLETE"]
    mf = {e["init_state_sha256"] for e in ok if e["method"].startswith("MF_")}
    mlp = {e["init_state_sha256"] for e in ok if e["method"].startswith("MLP_")}
    perm = {json.dumps(e["perm_stream"], sort_keys=True) for e in ok if e.get("perm_stream")}
    checks["all_smoke_units_complete"] = "PASS" if len(ok) == len(rows) == 7 else "FAIL"
    checks["MF_methods_share_initial_state"] = "PASS" if len(mf) == 1 and sum(e["method"].startswith("MF_") for e in ok) == 5 else "FAIL"
    checks["MLP_perm1_perm8_share_initial_state"] = "PASS" if len(mlp) == 1 and sum(e["method"].startswith("MLP_") for e in ok) == 2 else "FAIL"
    checks["perm_methods_share_derangement_stream"] = "PASS" if len(perm) == 1 else "FAIL"
    checks["evaluations_pass"] = "PASS" if all(e.get("evaluation", {}).get("status") == "PASS" for e in ok) and ok else "FAIL"
    checks["M1_first_step_from_W0"] = "PASS" if all(e["m1"]["first_step_W_start_is_zero"] for e in ok if "m1" in e) else "FAIL"
    # projection of the grid from the measured smoke costs (labelled proxy: the smoke condition is m = 32, K = 16 only)
    cost_of = {}
    for e in ok:
        cost_of[(e["method"], e["cap"])] = e["wall_seconds_unit_total"]
    proj = None
    if len(cost_of) == 7:
        lo = hi = 0.0
        n = collections.Counter()
        for row in U.load_rows():
            if row["block"] == "smoke":
                continue
            meth, cap = row["method"], row["cap"]
            n[row["block"]] += 1
            if meth == "MF_VCS_M1" and cap in ("CAL", "CAL_K16"):
                vals = [cost_of[(meth, str(c))] for c in C.M1_CAPS]
                lo, hi = lo + min(vals), hi + max(vals)
            else:
                v = cost_of[(meth, cap if meth == "MF_VCS_M1" else "NA")]
                lo, hi = lo + v, hi + v
        proj = {"units": dict(n), "wall_hours_sequential_low": lo / 3600, "wall_hours_sequential_high": hi / 3600,
                "basis": "per-unit wall seconds of the smoke unit with the same method (and cap); formal M1 rows use the min / max over caps "
                         "because their cap is CAL-selected; m = 8 and K = 8 / 32 rows are charged the m = 32, K = 16 smoke cost (proxy, "
                         "not measured); sequential single-process hours on the smoke node with its thread count; queue time excluded",
                "status": "PROJECTION_NOT_MEASUREMENT"}
    return {"schema": "v10_t03.smoke_summary.v1", "utc": C.utc_now(), "out_root": str(root), "units": rows, "checks": checks,
            "status": "PASS" if all(v == "PASS" for v in checks.values()) else "FAIL", "grid_cost_projection": proj,
            "environment": next((u["result"]["provenance"]["environment"] for u in units if u["result"] and u["result"].get("provenance", {}).get("environment")), None)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("smoke", "grid"))
    ap.add_argument("--output", type=Path)
    a = ap.parse_args(argv)
    root = C.out_root()
    res = smoke(root) if a.what == "smoke" else grid(root)
    path = a.output or (root / "aggregate" / ("T03_SMOKE_SUMMARY.json" if a.what == "smoke" else "T03_AGGREGATE.json"))
    res["aggregator_sha256"] = C.sha256_file(Path(__file__))
    C.write_json_atomic(path, res)
    print(json.dumps({"output": str(path), "status": res.get("status"), "checks": res.get("checks"),
                      "projection": res.get("grid_cost_projection"), "block_status": res.get("status") if a.what == "grid" else None}, indent=1))
    return 0 if a.what == "grid" or res["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
