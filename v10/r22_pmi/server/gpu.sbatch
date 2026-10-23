#!/usr/bin/env bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --signal=B:USR1@90
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R22-C (V10 PMI registration energy, evaluation only).  Usage:
#   gpu.sbatch run <unit.json>            one unit: CAL + DEV, J / PMI / EPT per registration (resumable output dir from the unit)
#   gpu.sbatch smoke <out dir> <unit.json>...
# Environment as the V10 BF1 / EPFL / app-round wrappers: env/bin/python, 4 BLAS threads, CUBLAS_WORKSPACE_CONFIG=:4096:8,
# TF32 override variables unset, torch defaults untouched.  Refuses to run with uncommitted changes in src or the R22-C
# code, wrappers and unit records (results / analysis files may change while jobs wait).
# A receipt (GPU, node, Slurm StartTime to end, exit code) goes to handoff/v10_r22_pmi_20261022/receipts/<job>.json.
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE PYTHONPATH
WT=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_r22c
REC=/home/infres/yinwang/CS_QMI/handoff/v10_r22_pmi_20261022/receipts
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
mode=${1:?run|smoke}; shift
start=''
for attempt in 1 2 3 4 5 6; do
  start=$(squeue -j "$SLURM_JOB_ID" -h -o '%S' 2>/dev/null | head -1 | tr -d '[:space:]' || true)
  case "$start" in 20[0-9][0-9]-*T*) break ;; *) start=''; sleep 10 ;; esac
done
export R22_START=${start:-$(date -u +%FT%T)} R22_MODE=$mode R22_ARGS="$*" R22_REC=$REC
export R22_GPU="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
export R22_COMMIT=$(git -C "$WT" rev-parse HEAD)
echo "host=$(hostname) gpu=${R22_GPU} start=${R22_START} job=${SLURM_JOB_ID} mode=${mode} args=$* commit=${R22_COMMIT}"
rc=0
if [ -n "$(git -C "$WT" status --porcelain -- src v10/r22_pmi/pmi_energy.py v10/r22_pmi/run_unit.py v10/r22_pmi/server v10/r22_pmi/units v10/epfl/r22_pmi/units)" ]; then
  echo "REFUSED: uncommitted changes in src or the R22-C code / units"; rc=3
else
  child=''
  forward() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi; }
  trap forward USR1 TERM
  set +e
  case "$mode" in
    run) "$PYTHON" -B -s "$WT/v10/r22_pmi/run_unit.py" run --unit "$1" & ;;
    smoke) out=$1; shift; "$PYTHON" -B -s "$WT/v10/r22_pmi/run_unit.py" smoke --out "$out" "$@" & ;;
    *) echo "unknown mode $mode"; exit 2 ;;
  esac
  child=$!
  wait "$child"; rc=$?
  if kill -0 "$child" 2>/dev/null; then wait "$child"; rc=$?; fi
  set -e
fi
export R22_RC=$rc
"$PYTHON" -B -s - <<'PY'
import datetime, json, os, time
from pathlib import Path
start = datetime.datetime.fromisoformat(os.environ["R22_START"]).replace(tzinfo=datetime.timezone.utc).timestamp()
r = {"job": os.environ["SLURM_JOB_ID"], "name": os.environ.get("SLURM_JOB_NAME"), "mode": os.environ["R22_MODE"],
     "args": os.environ["R22_ARGS"], "exit_code": int(os.environ["R22_RC"]), "gpu": os.environ.get("R22_GPU"),
     "node": os.environ.get("SLURMD_NODENAME"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
     "commit": os.environ["R22_COMMIT"], "start_unix": start, "end_unix": time.time()}
r["allocated_seconds"] = r["end_unix"] - r["start_unix"]
p = Path(os.environ["R22_REC"]); p.mkdir(parents=True, exist_ok=True)
(p / (r["job"] + ".json")).write_text(json.dumps(r, indent=2) + "\n")
PY
echo "end=$(date -u +%FT%TZ) exit=${rc}"
exit "$rc"
