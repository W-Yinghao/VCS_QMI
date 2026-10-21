"""Server base config, the package's 78-run pilot grid, and its registration in the package's global lock.

  base      after staging PASS: copy aigiqa/configs/base_alignment.json, replace ONLY the six staging fields
            (model.checkpoint/_sha256, model.med_config/_sha256, data_dir, output_dir); every scientific field stays as shipped.
  register  after G0 PASS on the partitions to be used: `aigiqa.cli make_grid --stage pilot` (package), check 78 runs,
            `vcs_common.experiment register` (package) BEFORE any training, `list-jobs`, then write JOBS.tsv and the
            per-run sbatch lines. Nothing is submitted here.
Package refs: aigiqa/README_CN.md section 3, shared/docs/EXPERIMENT_LOCK_CN.md section 4, SERVER_HANDOFF_CN.md section 7.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPLACED = ["model.checkpoint", "model.checkpoint_sha256", "model.med_config", "model.med_config_sha256", "data_dir", "output_dir"]
FAMILY_ORDER = ["task_only", "vcs_exact_text", "bce_text", "vcs_sampled8_text", "reference_nce_text", "corrected_cmim_both"]
LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]
SEEDS = [3101, 3102, 3103]
PARTITIONS = ["RTX6000PRO", "L40S", "H100"]
EXCLUDE = "node51,node53,node60"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def env_paths():
    vals = {}
    for line in Path(os.environ.get("AIG_PATHS_ENV", HERE / "paths.env")).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            for kk, vv in vals.items():
                v = v.replace("$" + kk, vv)
            vals[k] = v
    return vals


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def cmd_base(a, P):
    staging = json.loads(Path(P["AIG_DATA"], "STAGING_RECORD.json").read_text())
    if staging.get("status") != "PASS" or staging.get("dry_run"):
        sys.exit("staging record is not a PASS (non-dry) record")
    pkg_base = json.loads(Path(P["AIG_PKG"], "aigiqa/configs/base_alignment.json").read_text())
    base = copy.deepcopy(pkg_base)
    ma = staging["model_assets"]
    base["model"]["checkpoint"] = ma["ImageReward.pt"]["path"]
    base["model"]["checkpoint_sha256"] = ma["ImageReward.pt"]["sha256"]
    base["model"]["med_config"] = ma["med_config.json"]["path"]
    base["model"]["med_config_sha256"] = ma["med_config.json"]["sha256"]
    base["data_dir"] = P["AIG_PREPARED"]
    base["output_dir"] = str(Path(P["AIG_OUT"]) / "runs" / "AGIQA-3k" / "alignment")
    fa, fb = flatten(pkg_base), flatten(base)
    changed = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    if not set(changed) <= set(REPLACED):
        sys.exit(f"refuse: non-staging fields differ {sorted(set(changed) - set(REPLACED))}")
    out = Path(P["AIG_PROTOCOL"]) / "base_alignment.server.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as f:
        f.write(json.dumps(base, indent=2, ensure_ascii=False) + "\n")
    delta = {"package_base": str(Path(P["AIG_PKG"], "aigiqa/configs/base_alignment.json")),
             "package_base_sha256": sha256(Path(P["AIG_PKG"], "aigiqa/configs/base_alignment.json")),
             "server_base": str(out), "server_base_sha256": sha256(out),
             "replaced_fields": {k: {"package": fa.get(k), "server": fb.get(k)} for k in changed},
             "unchanged_scientific_fields": {k: v for k, v in fb.items() if k not in changed},
             "staging_record_sha256": sha256(Path(P["AIG_DATA"], "STAGING_RECORD.json"))}
    (out.parent / "BASE_CONFIG_DELTA.json").write_text(json.dumps(delta, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"server_base": str(out), "sha256": delta["server_base_sha256"], "replaced": changed}, indent=1))


def per_run_hours(g0_records):
    """Package budgeting rule (SERVER_HANDOFF section 7): measured epoch time x full candidate count, warm-up separately."""
    est = {}
    for rec in g0_records:
        if rec.get("status") != "PASS":
            continue  # failed/aborted G0 records carry no usable timing
        part = rec["slurm"]["SLURM_JOB_PARTITION"]
        fams = {}
        for fam, r in rec["families"].items():
            fit = r["timed_fit_epoch"]["seconds"]
            cal = r["cal_sized_inference"]["seconds"]
            build = r["model_build_seconds_incl_sha256"]
            warm = r.get("gaussian_warmup_epoch", {}).get("seconds", 0.0)
            ckpt = r["checkpoint_probe"]["bytes"]
            save = r["checkpoint_probe"]["save_seconds_target_fs"]
            # 20 epochs of FIT + CAL pass; worst case 20 best.pt rewrites at the save time measured on the run filesystem (AIG_OUT); model build plus
            # the lock's two full-asset checks (counted as 3 builds); Gaussian warm-up epoch for corrected_cmim_both.
            seconds = 20 * (fit + cal) + warm + build * 3 + 20 * save
            fams[fam] = {"fit_epoch_s": fit, "cal_pass_s": cal, "warmup_s": warm, "build_s": build, "ckpt_bytes": ckpt, "ckpt_save_s": save,
                         "estimated_run_hours": seconds / 3600}
        est[part] = fams
    return est


def cmd_register(a, P):
    proto = Path(P["AIG_PROTOCOL"])
    base = proto / "base_alignment.server.json"
    lock = Path(P["AIG_LOCK"])
    grid = proto / "aigiqa_alignment_grid"
    g0 = [json.loads(Path(p).read_text()) for p in a.g0]
    ok_parts = sorted({r["slurm"]["SLURM_JOB_PARTITION"] for r in g0 if r["status"] == "PASS"})
    env_lock = json.loads(Path(P["AIG_SERVER"], "ENV_LOCK.json").read_text())
    if not ok_parts:
        sys.exit("no G0 PASS record supplied")
    for r in g0:
        if r["status"] == "PASS" and r["runtime"]["torch"] != env_lock["torch"]["version"]:
            sys.exit("G0 ran in a different torch build than ENV_LOCK")
    env = {**os.environ}
    py = P["AIG_PY"]
    # 1) the package's own grid generator
    subprocess.run([py, "-m", "aigiqa.cli", "make_grid", "--base-config", str(base), "--output", str(grid), "--stage", "pilot"],
                   cwd=P["AIG_PKG"], env=env, check=True)
    with open(grid / "run_matrix.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    count = Counter(r["method"] for r in rows)
    expected = {"task_only": 3, **{fam: 15 for fam in FAMILY_ORDER if fam != "task_only"}}
    if len(rows) != 78 or dict(count) != expected:
        sys.exit(f"grid is not the package's 78-run pilot: {dict(count)}")
    for r in rows:
        if int(r["seed"]) not in SEEDS or (r["method"] != "task_only" and float(r["weight"]) not in LAMBDAS):
            sys.exit(f"unexpected seed/weight in {r}")
    # 2) global registration before any training (package EXPERIMENT_LOCK section 4), launch cwd = package root
    reg = subprocess.run([py, "-m", "vcs_common.experiment", "register", "--app", "aigiqa", "--grid", str(grid / "run_matrix.csv"),
                          "--output", str(lock), "--scope", "pilot", "--match", "bce_text,vcs_exact_text,vcs_sampled8_text",
                          "--primary-metric", "srcc", "--inputs", str(Path(P["AIG_PREPARED"]) / "test_inputs.csv")],
                         cwd=P["AIG_PKG"], env=env, check=True, capture_output=True, text=True)
    reg_out = json.loads(reg.stdout)
    jobs = json.loads(subprocess.run([py, "-m", "vcs_common.experiment", "list-jobs", "--lock", str(lock)], cwd=P["AIG_PKG"],
                                     env=env, check=True, capture_output=True, text=True).stdout)
    if len(jobs) != 78 or len({j["job_id"] for j in jobs}) != 78:
        sys.exit("lock does not hold 78 unique jobs")
    by_name = {Path(j["config"]).stem: j for j in jobs}
    est = per_run_hours(g0)
    worst = max(f["estimated_run_hours"] for part in est.values() for f in part.values())
    limit_h = max(2.0, math.ceil(3 * worst * 2) / 2)  # 3x the slowest measured family/type, >= 2 h, half-hour steps
    hh, mm = int(limit_h), int(round((limit_h - int(limit_h)) * 60))
    time_limit = f"{hh:02d}:{mm:02d}:00"
    parts = ",".join(p for p in PARTITIONS if p in ok_parts)
    order = sorted(rows, key=lambda r: (FAMILY_ORDER.index(r["method"]), int(r["seed"]), float(r["weight"])))
    lines, tsv = [], ["name\tjob_id\tfamily\tweight\tseed\trun_dir"]
    for r in order:
        name = Path(r["config"]).stem
        j = by_name[name]
        tsv.append(f"{name}\t{j['job_id']}\t{r['method']}\t{r['weight']}\t{r['seed']}\t{j['run_dir']}")
        lines.append(f"sbatch --job-name=aig_{name} --qos=normal --gres=gpu:1 --partition={parts} --exclude={EXCLUDE} "
                     f"--cpus-per-task=4 --mem=32G --time={time_limit} --nice=50 --no-requeue "
                     f"--output={P['AIG_LOGS']}/%x_%j.out {HERE}/train_job.sbatch {name}")
    (proto / "JOBS.tsv").write_text("\n".join(tsv) + "\n")
    (proto / "SUBMIT_LINES.sh").write_text("# 78 pilot runs; prerequisite for each: lock registered (this file) and nothing else.\n" + "\n".join(lines) + "\n")
    record = {"lock": str(lock), "lock_sha256": reg_out["lock_sha256"], "lock_file_sha256": sha256(lock), "jobs": len(jobs),
              "grid_matrix_sha256": sha256(grid / "run_matrix.csv"), "server_base_sha256": sha256(base),
              "family_counts": dict(count), "match_group": ["bce_text", "vcs_exact_text", "vcs_sampled8_text"],
              "primary_metric": "srcc", "scope": "pilot", "evaluation_inputs": str(Path(P["AIG_PREPARED"]) / "test_inputs.csv"),
              "g0_partitions_pass": ok_parts, "per_run_estimate_hours": est, "time_limit": time_limit,
              "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (proto / "LOCK_RECORD.json").write_text(json.dumps(record, indent=1) + "\n")
    ev = Path(P["AIG_EVIDENCE"]) / "lock"
    ev.mkdir(parents=True, exist_ok=True)
    for name in ("LOCK_RECORD.json", "JOBS.tsv", "SUBMIT_LINES.sh", "BASE_CONFIG_DELTA.json"):
        (ev / name).write_bytes((proto / name).read_bytes())
    (ev / "aigiqa_alignment_lock.json").write_bytes(lock.read_bytes())
    print(json.dumps({k: record[k] for k in ("lock", "lock_sha256", "jobs", "g0_partitions_pass", "time_limit")}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("base")
    r = sub.add_parser("register"); r.add_argument("--g0", nargs="+", required=True, help="G0_RESULT.json files")
    a = ap.parse_args()
    P = env_paths()
    {"base": cmd_base, "register": cmd_register}[a.cmd](a, P)


if __name__ == "__main__":
    main()
