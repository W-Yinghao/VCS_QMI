#!/usr/bin/env bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V10 app round 20261021: one package entry per allocation.  Submit only through submit.sh (sets --job-name/--time/--nice and
# records the job in the GPU-hour ledger).  Usage: gpu.sbatch <smoke|pair|diag> <resolved unit json> <new output dir>
# Environment and numeric settings are the original V10 BF1/EPFL wrappers' (env/bin/python, 4 BLAS threads,
# CUBLAS_WORKSPACE_CONFIG=:4096:8, TF32 override variables unset, torch defaults untouched).
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE PYTHONPATH
mode=${1:?smoke|pair|diag}; unit=${2:?unit}; out=${3:?out}
WT=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_appround
PKG=$WT/v10/app_round_20261021/package
LEDGER=/home/infres/yinwang/CS_QMI/handoff/v10_app_round_20261021/ledger
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
case "$mode" in smoke) script=server_smoke.py ;; pair) script=run_pair.py ;; diag) script=run_diagnostics.py ;; *) exit 2 ;; esac
sched_start=''
for attempt in 1 2 3 4 5 6; do
  sched_start=$(squeue -j "$SLURM_JOB_ID" -h -o '%S' 2>/dev/null | head -1 | tr -d '[:space:]' || true)
  case "$sched_start" in 20[0-9][0-9]-*T*) break ;; *) sched_start=''; sleep 10 ;; esac
done
export R21_START=${sched_start:-$(date -u +%FT%T)} R21_MODE=$mode R21_UNIT=$unit R21_OUT=$out R21_LEDGER=$LEDGER
export R21_GPU="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
export R21_HOST_COMMIT=$(git -C "$WT" rev-parse HEAD)
echo "host=$(hostname) gpu=${R21_GPU} start=${R21_START} job=${SLURM_JOB_ID} mode=${mode} unit=${unit} out=${out} host_commit=${R21_HOST_COMMIT}"
echo "precision_env CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG} TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE-<unset>} NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE-<unset>}"
rc=0
if [ -n "$(git -C "$WT" status --porcelain -- src scripts v10/app_round_20261021/package)" ]; then
  echo "REFUSED: host src/scripts or package files have uncommitted changes"; rc=3
else
  cd "$PKG"
  child=''
  forward() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi; }
  trap forward USR1 TERM
  set +e
  "$PYTHON" -B -s "$script" --unit "$unit" --repo "$WT" --out "$out" --device cuda &
  child=$!
  wait "$child"; rc=$?
  if kill -0 "$child" 2>/dev/null; then wait "$child"; rc=$?; fi
  set -e
fi
export R21_RC=$rc
"$PYTHON" -B -s - <<'PY'
import datetime, json, os, time
from pathlib import Path
start = datetime.datetime.fromisoformat(os.environ["R21_START"]).replace(tzinfo=datetime.timezone.utc).timestamp()
r = {"job": os.environ["SLURM_JOB_ID"], "name": os.environ.get("SLURM_JOB_NAME"), "mode": os.environ["R21_MODE"],
     "unit": os.environ["R21_UNIT"], "out": os.environ["R21_OUT"], "exit_code": int(os.environ["R21_RC"]),
     "gpu": os.environ.get("R21_GPU"), "node": os.environ.get("SLURMD_NODENAME"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
     "host_commit": os.environ["R21_HOST_COMMIT"], "start_unix": start, "end_unix": time.time()}
r["allocated_seconds"] = r["end_unix"] - r["start_unix"]
p = Path(os.environ["R21_LEDGER"]) / "receipts"; p.mkdir(parents=True, exist_ok=True)
(p / (r["job"] + ".json")).write_text(json.dumps(r, indent=2) + "\n")
PY
echo "end=$(date -u +%FT%TZ) exit=${rc}"
exit "$rc"
