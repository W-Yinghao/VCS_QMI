#!/usr/bin/env python3
"""M1 head-solve statistics per unit (review package §2 M1 row, §5, §6 item 5) from existing TRAIN_STEPS.jsonl records.

Per unit: cap, steps, cap-hit and converged fractions, relative gap (to the zero-head reference of the same batch) quantiles,
zero-reference steps, iterations, HVP calls and moment passes, stop reasons, effective precision fields, CAL selection.
The gap is the inner fixed-feature head problem of the current batch before the outer update: not an encoder certificate.
Usage: m1_stats.py OUT_DIR
"""
import csv, json, statistics, sys
from collections import Counter
from pathlib import Path

HO = Path("/home/infres/yinwang/CS_QMI/handoff")
GLOBS = ["v10_bf1_20261017/results/CAL/MF_VCS_M1/candidate_*", "v10_bf1_20261017/results/FINAL_DEV/MF_VCS_M1/seed_*",
         "v10_seed_replication_hw_20261019/results/m0_seed_*", "v10_seed_replication_pool_20261019/results/m0_seed_*",
         "v10_seed_ext_1006_1008_20261020/results/m0_seed_*",
         "v10_epfl_production_20261019/results/CAL/MF_VCS_M1/candidate_*", "v10_epfl_production_20261019/results/FINAL_DEV/MF_VCS_M1/seed_*",
         "v10_epfl_production_hw2_20261020/hw_controls/results/FINAL_DEV/MF_VCS_M1/seed_*",
         "v10_epfl_production_hw2_20261020/hw_replicates/results/FINAL_DEV/MF_VCS_M1/seed_*"]


def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def selected(d):
    sel = d.parent / "SELECTION.json"
    if d.name.startswith("candidate_") and sel.exists():
        s = json.load(open(sel)); s = s.get("selected") or s
        return str(s.get("order", s.get("candidate", {}).get("order") if isinstance(s.get("candidate"), dict) else s.get("candidate"))) == str(int(d.name.split("_")[1]))
    return ""


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); rows, reasons = [], {}
    for g in GLOBS:
        for d in sorted(p for p in HO.glob(g) if p.is_dir() and (p / "TRAIN_STEPS.jsonl").exists()):
            st = [json.loads(l) for l in open(d / "TRAIN_STEPS.jsonl")]
            st = [s for s in st if "m1_rule" in s]
            if not st:
                continue
            zero = [s for s in st if not s.get("m1_gap_ref_zero")]
            rel = [s["m1_gap_relative_to_zero"] for s in st if s.get("m1_gap_ref_zero") and s.get("m1_gap_relative_to_zero") is not None]
            it = [s["m1_iterations"] for s in st]
            env = sorted(d.glob("ENV_*.json")); e = json.load(open(env[-1])) if env else {}
            rc = Counter(s["m1_stop_reason"] for s in st); key = str(d.relative_to(HO)); reasons[key] = dict(rc)
            rows.append({"unit": key, "dataset": "EPFL" if "epfl" in key else "BF",
                         "complete": (d / "COMPLETE.json").exists(), "cal_selected": selected(d),
                         "cap": Counter(s["m1_max_iterations"] for s in st).most_common(1)[0][0], "steps": len(st),
                         "cap_hit_frac": rc.get("iteration_budget_exhausted", 0) / len(st),
                         "converged_frac": sum(bool(s.get("m1_converged")) for s in st) / len(st),
                         "rel_gap_median": q(rel, .5), "rel_gap_q90": q(rel, .9), "rel_gap_max": max(rel) if rel else None,
                         "zero_reference_steps": len(zero), "zero_reference_stop_reasons": dict(Counter(s["m1_stop_reason"] for s in zero)),
                         "iterations_median": statistics.median(it), "iterations_mean": sum(it) / len(it),
                         "hvp_calls_total": sum(s.get("m1_hvp_calls_total", 0) for s in st),
                         "moment_passes_total": sum(s.get("m1_moments_passes_total", 0) for s in st),
                         "m1_rule": st[0]["m1_rule"], "relative_tolerance": st[0]["m1_relative_tolerance"], "m1_dtype": st[0]["m1_dtype"],
                         "require_converged": st[0]["m1_require_converged"], "gpu": e.get("gpu_name"),
                         "float32_matmul_precision": e.get("float32_matmul_precision"), "cudnn_conv_fp32_precision": e.get("cudnn_conv_fp32_precision"),
                         "deterministic_algorithms": e.get("deterministic_algorithms")})
    with open(out / "M1_GAP_STATS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "M1_STOP_REASONS.json").write_text(json.dumps(reasons, indent=1) + "\n")
    for r in rows:
        print(f"{r['unit'][-58:]:58s} cap{r['cap']:>4} sel={str(r['cal_selected'])[:1]} hit={r['cap_hit_frac']:.3f} conv={r['converged_frac']:.3f} "
              f"relgap_med={r['rel_gap_median'] if r['rel_gap_median'] is None else round(r['rel_gap_median'], 4)} zero={r['zero_reference_steps']} {r['gpu'] or ''}")


if __name__ == "__main__":
    main()
