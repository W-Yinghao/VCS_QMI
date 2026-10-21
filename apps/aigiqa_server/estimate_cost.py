"""GPU-time and disk estimate for the 78-run AGIQA-3K alignment pilot (package rule: measured epoch time x full
candidate count; CAL pass, Gaussian warm-up and checkpoint I/O counted separately; failures/reruns not included).

  python estimate_cost.py                     pre-G0 planning numbers from the stated assumptions
  python estimate_cost.py G0_RESULT_*.json    per GPU-type stratum from the G0 measurements
"""
from __future__ import annotations

import json
import sys

RUNS = {"task_only": 3, "reference_nce_text": 15, "bce_text": 15, "vcs_exact_text": 15, "vcs_sampled8_text": 15, "corrected_cmim_both": 15}
EPOCHS, SELECTED = 20, 18  # 6 families x 3 seeds keep their CAL-selected checkpoint after the TEST predictions
WRITE_BPS = 100e6          # assumed run-filesystem (AIG_OUT, home) write rate, pre-G0 only; G0 measures it
PRE_G0 = {"fit_images": 2086, "cal_images": 447, "batch": 5, "step_s": (0.10, 0.25), "infer_batch_s": (0.03, 0.06), "build_s": 60,
          "warmup_epoch_s": (30, 80), "ckpt_bytes": 1.84e9, "max_ckpt_writes": 20}


def run_seconds(fit_s, cal_s, warm_s, build_s, ckpt_bytes, writes=20, save_s=None):
    save = ckpt_bytes / WRITE_BPS if save_s is None else save_s
    return EPOCHS * (fit_s + cal_s) + warm_s + 3 * build_s + writes * save


def table(per_family):
    total = sum(RUNS[f] * h for f, h in per_family.items())
    return {"per_run_hours": {f: round(h, 3) for f, h in per_family.items()}, "matrix_gpu_hours_if_all_on_this_type": round(total, 1),
            "wall_hours_on_8_gpus": round(total / 8, 1)}


def main():
    out = {}
    if len(sys.argv) == 1:
        p = PRE_G0
        steps, cal_batches = p["fit_images"] // p["batch"], -(-p["cal_images"] // p["batch"])
        for label, i in (("optimistic", 0), ("conservative", 1)):
            fam = {}
            for f in RUNS:
                warm = p["warmup_epoch_s"][i] if f == "corrected_cmim_both" else 0
                fam[f] = run_seconds(steps * p["step_s"][i], cal_batches * p["infer_batch_s"][i], warm, p["build_s"], p["ckpt_bytes"]) / 3600
            out[f"pre_G0_{label}_L40S_assumption"] = table(fam)
        ckpt = p["ckpt_bytes"]
        out["assumptions"] = p
    else:
        ckpt = 0
        for path in sys.argv[1:]:
            rec = json.load(open(path))
            part = rec["slurm"]["SLURM_JOB_PARTITION"]
            if rec.get("status") != "PASS":
                out[f"{part}_{rec['slurm'].get('SLURM_JOB_ID')}"] = {"status": rec.get("status"), "skipped": "not a PASS record"}
                continue
            fam = {}
            for f, r in rec["families"].items():
                ckpt = max(ckpt, r["checkpoint_probe"]["bytes"])
                fam[f] = run_seconds(r["timed_fit_epoch"]["seconds"], r["cal_sized_inference"]["seconds"],
                                     r.get("gaussian_warmup_epoch", {}).get("seconds", 0.0), r["model_build_seconds_incl_sha256"],
                                     r["checkpoint_probe"]["bytes"], save_s=r["checkpoint_probe"].get("save_seconds_target_fs")) / 3600
            out[f"{part}_{rec['slurm'].get('SLURM_JOB_ID')}"] = table(fam) | {"status": rec["status"]}
    out["disk"] = {"best_pt_bytes_per_run": ckpt, "all_78_after_training_GB": round(78 * ckpt / 1e9, 1),
                   "kept_18_selected_after_TEST_predictions_GB": round(SELECTED * ckpt / 1e9, 1),
                   "deletable_60_non_selected_after_TEST_predictions_GB": round(60 * ckpt / 1e9, 1),
                   "peak_during_training_GB": round(78 * ckpt / 1e9, 1),
                   "note": "seal/check-release re-hash every registered best.pt, so none may be deleted before the release's TEST predictions are done"}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
