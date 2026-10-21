#!/usr/bin/env python
"""GPU-time and disk estimate for the KD CAL matrix and the 3-seed final phase.

Package method (INSTALL_CN §5, SERVER_HANDOFF_CN §七): measured per-family FIT epoch wall time x 240 epochs x the
expanded candidate count (CAL inference, checkpoint I/O and history writes are inside the measured epoch-to-epoch
wall time; startup + first epoch measured separately); disk = bytes per kept file x candidates. Timings are
reported per GPU stratum and never mixed. Without --g0 it prints PROVISIONAL analytic disk figures and an
explicitly unmeasured time range (the package forbids fixed GPU-hour commitments before measurement).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

KD_ROOT = Path(os.environ.get("KD_ROOT", "/home/infres/yinwang/CS_QMI/outputs_apps/kd_20261020"))
EPOCHS = 240
FINAL_SEEDS = 3
KEPT = ("best.pt", "best_cal_predictions.csv", "best_cal_metrics.json", "run.json", "resolved_config.json")
MB = 1e6

# Analytic (pre-G0) sizes from parameter counts: WRN-16-2 student 703,284 params (+1,824 BN running stats),
# SGD momentum of the same size; heads <= 0.4 MB; official CRD memory banks 2 x 45,000 x 128 float32 = 46.08 MB.
ANALYTIC_CKPT_MB = {"default": 5.9, "crd_original": 52.0, "dkd_crd_uniform": 52.0}
ANALYTIC_CAL_CSV_MB = 11.0      # 5,000 rows x 100 class probabilities as JSON floats (~2.2 kB/row)
ANALYTIC_HISTORY_MB = 0.8       # 240 epoch records


def counts_from_grid(grid):
    rows = json.loads(Path(grid).read_text())["rows"]
    return collections.Counter(r["method"] for r in rows)


def analytic(counts):
    per = {}
    for fam, n in counts.items():
        ck = ANALYTIC_CKPT_MB.get(fam, ANALYTIC_CKPT_MB["default"])
        kept = ck + ANALYTIC_CAL_CSV_MB + ANALYTIC_HISTORY_MB
        per[fam] = {"runs": n, "kept_MB_per_run": round(kept, 1), "peak_MB_per_run_with_last_pt": round(kept + ck, 1)}
    cal_kept = sum(v["runs"] * v["kept_MB_per_run"] for v in per.values())
    cal_peak_all = sum(v["runs"] * v["peak_MB_per_run_with_last_pt"] for v in per.values())
    final_kept = sum(FINAL_SEEDS * (v["kept_MB_per_run"] + 2 * ANALYTIC_CAL_CSV_MB) for v in per.values())
    return {"status": "PROVISIONAL_ANALYTIC_NO_MEASUREMENT", "per_family": per,
            "cal_matrix_runs": sum(counts.values()), "cal_matrix_kept_GB": round(cal_kept / 1e3, 2),
            "cal_matrix_upper_bound_if_every_last_pt_existed_GB": round(cal_peak_all / 1e3, 2),
            "final_phase_runs": FINAL_SEEDS * len(counts), "final_phase_kept_GB_incl_TEST_csv": round(final_kept / 1e3, 2),
            "time": analytic_time(counts)}


# Unmeasured per-epoch wall-time ranges (seconds): batch-64 32x32 training is launch-latency bound (teacher forward +
# student forward/backward + CAL inference each epoch). Official CRD adds per-row CPU negative sampling
# (64 rows x choice of 16,384 of 45,000 ids per batch); M1 adds 30 inner solver iterations per batch.
ASSUMED_EPOCH_SECONDS = {"default": (10, 35), "crd_original": (25, 60), "dkd_crd_uniform": (25, 60), "dkd_vcs_m1": (12, 40)}


def analytic_time(counts):
    lo = hi = 0.0
    per = {}
    for fam, n in counts.items():
        a, b = ASSUMED_EPOCH_SECONDS.get(fam, ASSUMED_EPOCH_SECONDS["default"])
        per[fam] = [round(a * EPOCHS / 3600, 2), round(b * EPOCHS / 3600, 2)]
        lo += n * a * EPOCHS / 3600
        hi += n * b * EPOCHS / 3600
    final_lo = sum(FINAL_SEEDS * v[0] for v in per.values())
    final_hi = sum(FINAL_SEEDS * v[1] for v in per.values())
    return {"status": "UNMEASURED_ORDER_OF_MAGNITUDE", "assumed_epoch_seconds": ASSUMED_EPOCH_SECONDS,
            "hours_per_run_range": per, "cal_matrix_gpu_hours_range": [round(lo), round(hi)],
            "final_phase_gpu_hours_range": [round(final_lo), round(final_hi)],
            "teacher_wrn40_2_hours_range": [round(10 * EPOCHS / 3600, 1), round(35 * EPOCHS / 3600, 1)],
            "note": "no GPU-hour commitment before G0 (INSTALL_CN §5); replace with estimate_cost.py --g0 <G0_SUMMARY.json ...>"}


def measured(paths, counts):
    strata = {}
    for path in paths:
        s = json.loads(Path(path).read_text())
        fam_w1, fam_w100 = s.get("families", {}).get("w1", {}), s.get("families", {}).get("w100", {})
        if not fam_w1:
            continue
        stratum = s.get("slurm_partition") or Path(path).parent.name
        gpu = (s.get("stages", {}).get("env", {}).get("runtime", {}).get("gpus") or [{}])[0].get("name")
        per, missing = {}, []
        for fam, n in counts.items():
            row = fam_w1.get(fam) or fam_w100.get(fam)
            if not row or not row.get("estimated_240_epoch_seconds"):
                missing.append(fam)
                continue
            est = row["estimated_240_epoch_seconds"]
            alt = (fam_w100.get(fam) or {}).get("estimated_240_epoch_seconds")
            files = row.get("file_bytes") or {}
            hist = (row.get("history_bytes_per_epoch") or 0) * EPOCHS
            kept = sum(files.get(k, 0) for k in KEPT) + hist
            per[fam] = {"runs": n, "est_hours_per_run": round(est / 3600, 3),
                        "est_hours_per_run_w100": round(alt / 3600, 3) if alt else None,
                        "steady_epoch_wall_seconds": row.get("steady_epoch_wall_seconds"),
                        "cuda_peak_allocated_GB": round((row.get("cuda_peak_allocated_bytes") or 0) / 1e9, 2),
                        "kept_MB_per_run": round(kept / MB, 1), "last_pt_MB": round(files.get("last.pt", 0) / MB, 1)}
        cal_h = sum(v["runs"] * v["est_hours_per_run"] for v in per.values())
        final_h = sum(FINAL_SEEDS * v["est_hours_per_run"] for v in per.values())
        strata[stratum] = {"gpu": gpu, "source": str(path), "per_family": per, "missing_families": missing,
                           "cal_matrix_gpu_hours_if_all_on_this_type": round(cal_h, 1),
                           "final_phase_gpu_hours_if_all_on_this_type": round(final_h, 1),
                           "cal_matrix_kept_GB": round(sum(v["runs"] * v["kept_MB_per_run"] for v in per.values()) / 1e3, 2)}
    return {"status": "MEASURED_PER_STRATUM", "strata": strata,
            "note": "per-stratum figures are not mixed; the actual matrix cost is the sum over the GPUs the jobs land on (ledger)"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", default=str(KD_ROOT / "protocol" / "kd_cal_grid" / "grid_manifest.json"))
    p.add_argument("--g0", nargs="*", default=[])
    p.add_argument("--out")
    args = p.parse_args()
    counts = counts_from_grid(args.grid)
    result = measured(args.g0, counts) if args.g0 else analytic(counts)
    result["family_run_counts"] = dict(counts)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
