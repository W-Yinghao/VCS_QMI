#!/usr/bin/env bash
#SBATCH --job-name=v10e_fin
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --exclude=node51,node60
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# EPFL production final seed for one family (after its CAL selection and a successful CAL exit receipt): 3000 steps, CAL at
# every checkpoint, DEV (74 x 10) only at its own CAL-selected checkpoint.  Usage (from the job plan):
#   V10_EPFL_SOURCE=<frozen source> V10_EPFL_RELEASE=<bundle>/RELEASE.json V10_EPFL_EXPECT_RELEASE=<sha256> \
#     sbatch --export=ALL --job-name=v10e_fin_<fam>_<seed> --time=<limit> <source>/slurm/v10_epfl/prod_final.sbatch '<family>' <seed>
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
: "${SLURM_JOB_ID:?Slurm required}"
: "${V10_EPFL_SOURCE:?frozen source required}"
: "${V10_EPFL_RELEASE:?release path required}"
: "${V10_EPFL_EXPECT_RELEASE:?release sha256 pin required}"
family=${1:?family}; seed=${2:?seed}
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
# Allocated time runs from the Slurm StartTime (UTC, TZ above); retried because a busy controller can miss one query.
sched_start=''
for attempt in 1 2 3 4 5 6; do
  sched_start=$(squeue -j "$SLURM_JOB_ID" -h -o '%S' 2>/dev/null | head -1 | tr -d '[:space:]' || true)
  case "$sched_start" in 20[0-9][0-9]-*T*) break ;; *) sched_start=''; sleep 10 ;; esac
done
: "${sched_start:?Slurm StartTime unavailable: no cost accounting, refusing to run}"
export V10_EPFL_JOB_START
V10_EPFL_JOB_START=$("$PYTHON" -c 'import sys,datetime; print(datetime.datetime.fromisoformat(sys.argv[1]).replace(tzinfo=datetime.timezone.utc).timestamp())' "$sched_start")
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) job=${SLURM_JOB_ID} family=${family} seed=${seed} source=${V10_EPFL_SOURCE} commit=$(git -C "$V10_EPFL_SOURCE" rev-parse HEAD) release=${V10_EPFL_EXPECT_RELEASE} slurm_start_epoch=${V10_EPFL_JOB_START}"
cd "$V10_EPFL_SOURCE"
child=''
forward_signal() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap forward_signal USR1 TERM
"$PYTHON" -I scripts/v10_epfl/run.py final "$family" --seed "$seed" --release "$V10_EPFL_RELEASE" &
child=$!
set +e
wait "$child"; rc=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; rc=$?; fi
export V10_EPFL_EXIT_CODE=$rc
"$PYTHON" - <<'PY'
import json, os, time
from pathlib import Path
p = Path(os.environ["V10_EPFL_RELEASE"]).resolve().parent / "exits"; p.mkdir(parents=True, exist_ok=True)
r = {"job": os.environ["SLURM_JOB_ID"], "exit_code": int(os.environ["V10_EPFL_EXIT_CODE"]), "start_unix": float(os.environ["V10_EPFL_JOB_START"]),
     "end_unix": time.time(), "release_sha256": os.environ["V10_EPFL_EXPECT_RELEASE"]}
r["allocated_seconds"] = r["end_unix"] - r["start_unix"]
t = p / ("." + r["job"] + ".tmp"); t.write_text(json.dumps(r, indent=2) + "\n"); t.replace(p / (r["job"] + ".json"))
PY
receipt_rc=$?
if [ "$rc" -eq 0 ] && [ "$receipt_rc" -ne 0 ]; then rc=$receipt_rc; fi
echo "end=$(date -u +%FT%TZ) exit=${rc}"
exit "$rc"
