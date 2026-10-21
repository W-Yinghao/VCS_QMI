#!/usr/bin/env python
"""Emit one sbatch line per globally locked KD candidate.

  --lock L [--precomputed P]  after `register`: job IDs from the lock (checked with vcs_common.experiment.check);
                               with P, also verifies the IDs equal the pre-registration dry run.
  --precomputed P             before `register`: PROVISIONAL lines (IDs from the dry run, lock path = planned path).
  --g0 S1 S2 ...              G0_SUMMARY.json files: per-family --time = 3 x the maximum estimated 240-epoch
                              wall time over all measured GPU strata (+15 min, rounded up to 15 min, 1 h <= t <= 24 h),
                              because a job may start on any of the three partitions. Without --g0: provisional limits.
Order is risk-first (EXPERIMENT_LOCK §6: any FAILED candidate blocks seal, so likely failures should surface early):
  1 lambda=100 candidates, 2 fixed ITRD/CRD references, 3 lambda=10, 4 remaining M1, 5 the rest.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

KD_ROOT = Path(os.environ.get("KD_ROOT", "/home/infres/yinwang/CS_QMI/outputs_apps/kd_20261020"))
WRAPPER = "/home/infres/yinwang/CS_QMI/ssl_pilot_app_kd/apps/kd_server/sbatch/kd_lock_train.sbatch"
PLANNED_LOCK = str(KD_ROOT / "protocol" / "kd_cal_lock.json")
PROVISIONAL = {"default": "12:00:00", "crd_original": "18:00:00", "dkd_crd_uniform": "18:00:00", "dkd_vcs_m1": "18:00:00"}
COMMON = ("--qos=normal --partition={partitions} --exclude=node51,node53,node60 --gres=gpu:1 "
          "--cpus-per-task=8 --mem=24G --no-requeue --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out")


def hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:00"


def g0_limits(paths):
    worst = {}
    for path in paths:
        summary = json.loads(Path(path).read_text())
        for table in summary.get("families", {}).values():
            for family, row in table.items():
                est = row.get("estimated_240_epoch_seconds")
                if est:
                    worst[family] = max(worst.get(family, 0.0), float(est))
    limits = {}
    for family, est in worst.items():
        t = 3 * est + 900
        t = math.ceil(t / 900) * 900
        limits[family] = hms(min(max(t, 3600), 24 * 3600))
    return limits, worst


def stem_coefficient(stem):
    """CAL grid stems are <family>_w<coef>_seed<S>; final-grid stems <family>_seed<S> carry no coefficient."""
    try:
        return float(stem.rsplit("_seed", 1)[0].rsplit("_w", 1)[1])
    except (IndexError, ValueError):
        return None


def priority(job):
    c, fam = job["coefficient"], job["family"]
    if c is None:
        return (5, fam, 0.0)
    if c == 100:
        return (1, fam)
    if fam in ("itrd_reference", "crd_original", "dkd_crd_uniform"):
        return (2, fam)
    if c == 10:
        return (3, fam)
    if fam == "dkd_vcs_m1":
        return (4, c)
    return (5, fam, c)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock")
    p.add_argument("--precomputed")
    p.add_argument("--g0", nargs="*", default=[])
    p.add_argument("--nice", type=int, default=50)
    p.add_argument("--prefix", default="kd_cal")
    p.add_argument("--partitions", default="RTX6000PRO,L40S,H100", help="drop a GPU type whose G0 failed")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    pre = json.loads(Path(args.precomputed).read_text()) if args.precomputed else None
    if args.lock:
        from vcs_common import experiment as ex
        lock = ex.check(args.lock)
        by_id = {j["job_id"]: j for j in lock["jobs"]}
        jobs = []
        coef = {Path(j["config_path"]).stem: j for j in (pre["jobs"] if pre else [])}
        for job_id, j in by_id.items():
            stem = Path(j["config_path"]).stem
            coefficient = coef[stem]["coefficient"] if stem in coef else stem_coefficient(stem)
            jobs.append({"job_id": job_id, "family": j["family"], "coefficient": coefficient, "config_stem": stem})
        if pre and {j["job_id"] for j in pre["jobs"]} != set(by_id):
            raise SystemExit("registered job IDs differ from the precomputed dry run - use the lock's IDs and investigate")
        lock_path, status = str(Path(args.lock).resolve()), "REGISTERED"
    elif pre:
        jobs, lock_path, status = pre["jobs"], PLANNED_LOCK, "PROVISIONAL_BEFORE_REGISTRATION"
    else:
        raise SystemExit("need --lock and/or --precomputed")
    limits, worst = g0_limits(args.g0) if args.g0 else ({}, {})
    lines = [f"# {status}: {len(jobs)} lines; prerequisite: lock registered (S4) after teacher PASS and G0 PASS; "
             f"time limits {'from G0 (3x max stratum estimate)' if limits else 'PROVISIONAL (no G0 yet)'}"]
    for job in sorted(jobs, key=priority):
        fam = job["family"]
        t = limits.get(fam) or PROVISIONAL.get(fam, PROVISIONAL["default"])
        name = f"{args.prefix}_{job['config_stem'].rsplit('_seed', 1)[0]}"
        lines.append(f"sbatch --job-name={name} {COMMON.format(partitions=args.partitions)} --time={t} --nice={args.nice} {WRAPPER} {lock_path} {job['job_id']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": status, "lines": len(jobs), "out": args.out,
                      "time_limits": limits or "provisional", "worst_est_240_seconds": worst}, indent=1))


if __name__ == "__main__":
    main()
