#!/usr/bin/env bash
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nice=0
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# BF1_CAPACITY_EXT_K8_K32: one unit per job.  --job-name=v10k_<fam>_k<K>_<seed> and --time (perm1/M0 01:30:00, M1 03:00:00) are
# given per unit on the sbatch line (bundle SUBMISSION.json); run.py refuses a job whose name, partition or time limit differs.
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE PYTHONPATH
: "${V10_CAPACITY_FREEZE:?sha256 of the bundle FREEZE.json is required (--export=ALL,V10_CAPACITY_FREEZE=...)}"
# The frozen BF1 scientific tree pins (as BF1 gpu.sbatch exported them for the K16 runs).
export V10_EXPECT_CODE=e08ee2c176475e2b V10_SUBMIT_COMMIT=b8f59ea825dc7a2064457d47a628bef09ca4e185
WT=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_capacity
OUT=/projects/EEG-foundation-model/yinghao/outputs_v10/capacity_ext_20261019
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
mkdir -p "$OUT"
cd "$OUT"   # neutral working directory: never a frozen tree or a git worktree
V10K_BATCH_START=$(date +%s.%N)
export V10K_BATCH_START
q=''
for attempt in 1 2 3; do
  q=$(squeue -j "$SLURM_JOB_ID" -h -o '%S|%l' 2>/dev/null || true)
  if [ -n "$q" ]; then break; fi
  sleep 5
done
export V10K_SCHED_START_RAW="${q%%|*}" V10K_TIME_LIMIT_RAW="${q##*|}"
child=''
forward_signal() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap forward_signal USR1 TERM
"$PYTHON" -u "$WT/scripts/v10_capacity/run.py" "${1:?unit key}" &
child=$!
set +e
wait "$child"
rc=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; rc=$?; fi
used=$(squeue -j "$SLURM_JOB_ID" -h -o '%M' 2>/dev/null)
V10K_USED_SAMPLE_NODE=$(date +%s.%N)
V10K_EXIT_CODE=$rc V10K_CONTROLLER_USED_RAW="$used" V10K_USED_SAMPLE_NODE=$V10K_USED_SAMPLE_NODE \
  "$PYTHON" "$WT/scripts/v10_capacity/receipt.py" "$1"
receipt_rc=$?
if [ "$rc" -eq 0 ] && [ "$receipt_rc" -ne 0 ]; then rc=$receipt_rc; fi
exit "$rc"
