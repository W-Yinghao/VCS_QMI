"""Emit sbatch lines (one per JOBS row) for app_lines_feed.sh. Usage: make_lines.py CODE_DIR JOBS_REL [--grad-check]"""
import csv
import sys
from pathlib import Path

WRAP = "/home/infres/yinwang/CS_QMI/ssl_pilot_app_r22/apps/r22/server/run_one.sbatch"
RES = {"kd": "--cpus-per-task=8 --mem=24G --time=04:30:00", "aigiqa": "--cpus-per-task=4 --mem=32G --time=02:00:00"}
SMOKE = {"kd": "--cpus-per-task=8 --mem=24G --time=00:40:00", "aigiqa": "--cpus-per-task=4 --mem=32G --time=00:50:00"}
code, rel = sys.argv[1], sys.argv[2]
extra = " --grad-check" if "--grad-check" in sys.argv[3:] else ""
for r in csv.DictReader((Path(code) / rel).open(newline="")):
    res = (SMOKE if extra else RES)[r["app"]]
    print(f"sbatch --job-name={r['job_name']} --qos=normal --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 "
          f"--gres=gpu:1 {res} --nice={0 if extra else 50} --no-requeue --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out "
          f"{WRAP} {code} {rel}{extra}")
