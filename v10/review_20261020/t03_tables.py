#!/usr/bin/env python3
"""T03 deliverables of the review package (§3.4, §6 item 8) from the completed formal records at v10/t03 (6ccd4822).

T03_BIAS_COMPONENTS.csv: one row per formal run (MAIN/ALGO/CAP): EVAL-S, population MSE, EVAL-J_pop of the selected critic,
  with condition and data-replicate keys for pairing. Uses S - J_pop(T) = E_M[(T - eta)^2] (identity checked in the records).
T03_M1_PRECISION_DISAGREEMENTS.json: M1 fp32-vs-fp64 stop-decision diagnostic per condition / cap / direction, distance of the
  fp32 gap to the threshold, and the selected checkpoint's head-gap (fp32, fp64, polish). Training-time gaps are the batch
  problem before the outer update; checkpoint gaps are after it; the two are not interchangeable.
Usage: t03_tables.py OUT_DIR
"""
import csv, gzip, json, sys
from collections import defaultdict
from pathlib import Path

FORMAL = Path("/home/infres/yinwang/CS_QMI/ssl_pilot_v10_t03/v10/T03/formal/runs")
OUTPUTS = Path("/projects/EEG-foundation-model/yinghao/outputs_v10/T03/runs")


def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rows = []; agg = defaultdict(lambda: {"triggered": 0, "disagree": 0, "fp32_converged_fp64_not": 0,
                                          "fp32_not_fp64_converged": 0, "dist": []})
    sel_gaps = []
    for d in sorted(FORMAL.glob("T03_*_*")):
        if d.name.startswith(("T03_CAL", "T03_SMK")) or not (d / "evaluator_only/evaluation.json").exists():
            continue
        u = json.load(open(d / "result.json"))["unit_row"]; e = json.load(open(d / "evaluator_only/evaluation.json"))
        m = e["selected_metrics"]
        rows.append({"unit_id": u["unit_id"], "block": u["block"], "family": u["family"], "m": u["m"], "rho": u["rho"],
                     "method": u["method"], "K": u["K"], "lr": u["lr"], "cap": u["cap"], "data_replicate": u["data_root_seed"],
                     "algo_seed": u["algo_seed"], "selected_step": e["selected"]["step"], "S": m["S"], "J_pop": m["J_pop"],
                     "population_MSE": m["pop_mse"], "EVAL_UJ": m["EVAL_UJ"], "EVAL_minus_S": m["EVAL_UJ_minus_S"],
                     "EVAL_minus_J_pop": m["EVAL_UJ_minus_J_pop"], "J_pop_minus_S": m["J_pop_minus_S"],
                     "identity_check": m["identity_check"]})
        if u["method"] != "MF_VCS_M1":
            continue
        s = json.load(gzip.open(OUTPUTS / d.name / "train_steps.json.gz"))
        key = (u["block"], u["family"], u["m"], u["rho"], u["cap"])
        a = agg[key]
        for i, trig in enumerate(s["m1_fp64_triggered"]):
            if not trig:
                continue
            a["triggered"] += 1
            if s["m1_fp64_decisions_agree"][i]:
                continue
            a["disagree"] += 1
            a["fp32_converged_fp64_not" if s["m1_converged"][i] else "fp32_not_fp64_converged"] += 1
            tol = s["m1_tolerance_effective"][i]
            a["dist"].append((s["m1_gap_final"][i] - tol) / tol if tol else None)
        hg = m.get("head_gap") or {}
        sel_gaps.append({"unit_id": u["unit_id"], "condition": dict(zip(("block", "family", "m", "rho", "cap"), key)),
                         "selected_step": e["selected"]["step"], "head_gap": hg})
    with open(out / "T03_BIAS_COMPONENTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    by_cond, tot = [], {"triggered": 0, "disagree": 0, "fp32_converged_fp64_not": 0, "fp32_not_fp64_converged": 0}
    by_rho = defaultdict(lambda: {"triggered": 0, "disagree": 0})
    for key, a in sorted(agg.items()):
        dist = [x for x in a["dist"] if x is not None]
        by_cond.append({"block": key[0], "family": key[1], "m": key[2], "rho": key[3], "cap": key[4],
                        **{k: a[k] for k in tot},
                        "relative_distance_of_fp32_gap_to_threshold": {"median": q(dist, .5), "q10": q(dist, .1), "q90": q(dist, .9),
                                                                       "min": min(dist) if dist else None, "max": max(dist) if dist else None}})
        for k in tot: tot[k] += a[k]
        by_rho[key[3]]["triggered"] += a["triggered"]; by_rho[key[3]]["disagree"] += a["disagree"]
    (out / "T03_M1_PRECISION_DISAGREEMENTS.json").write_text(json.dumps({
        "schema": "v10.review.t03_m1_precision_disagreements.v1", "source": str(FORMAL), "step_records": str(OUTPUTS),
        "definition": "per triggered training step, the fp32 stop decision (gap_final <= tolerance) is re-made in fp64 on the same input; "
                      "fp32_converged_fp64_not = fp32 reported converged but fp64 does not (judged met when not); fp32_not_fp64_converged = "
                      "the reverse. Distance = (fp32 gap_final - tolerance) / tolerance. The fp64 gap values themselves are not stored "
                      "per step (only the agreement flag), so they cannot be tabulated.",
        "denominator_note": "the fraction is over diagnostic-triggered steps only, not over all training steps",
        "totals": tot, "by_rho": by_rho, "by_condition": by_cond, "selected_checkpoint_head_gap": sel_gaps}, indent=1) + "\n")
    print(json.dumps({"bias_rows": len(rows), "m1_runs": len(sel_gaps), "totals": tot, "by_rho": by_rho}))


if __name__ == "__main__":
    main()
