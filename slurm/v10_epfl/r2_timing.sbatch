#!/usr/bin/env bash
#SBATCH --job-name=v10e_r2time
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --exclude=node51,node60
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V10 EPFL RuLSIF_exact_v2 measured cost on L40S (scripts/v10_epfl/rulsif_v2_timing.py): build, complete fit, checkpoint and one
# complete CAL pass (100 registrations), assembled to candidate / CAL family / final seed.  Submit only after r2_gpu_gate has PASSED.
# Runs from the frozen gate source tree; never edit it while this job is queued or running:
#   S=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_epfl_rulsif2_gate3
#   V10_EPFL_SOURCE=$S V10_EPFL_EXPECT_COMMIT=$(git -C $S rev-parse HEAD) \
#   V10_EPFL_RECORD_SHA256=$(sha256sum $S/v10/epfl/EPFL_ACTIVATION_RECORD.json | cut -c1-64) \
#   V10_EPFL_GROUP_SHA256=$(sha256sum $S/v10/epfl/EPFL_GROUP_RECORD_V2_FROZEN.json | cut -c1-64) \
#     sbatch --export=ALL --job-name=v10e_r2time<T> $S/slurm/v10_epfl/r2_timing.sbatch --centres <T>   # only after r2_gpu_gate PASSED
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
export V10_EPFL_OUT=${V10_EPFL_OUT:-/home/infres/yinwang/CS_QMI/outputs/v10_epfl_rulsif_v2}
: "${SLURM_JOB_ID:?Slurm required}"
: "${V10_EPFL_SOURCE:?frozen gate source required}"
: "${V10_EPFL_EXPECT_COMMIT:?commit pin required}"
: "${V10_EPFL_RECORD_SHA256:?GATE_CHECKS activation record pin required}"
: "${V10_EPFL_GROUP_SHA256:?group record pin required}"
R=$V10_EPFL_SOURCE
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) job=${SLURM_JOB_ID} partition=${SLURM_JOB_PARTITION:-NA} commit=$(git -C "$R" rev-parse HEAD) expect=${V10_EPFL_EXPECT_COMMIT} record_pin=${V10_EPFL_RECORD_SHA256} group_pin=${V10_EPFL_GROUP_SHA256} out=${V10_EPFL_OUT}"
mkdir -p "$V10_EPFL_OUT"
cd "$R"
child=''
forward_signal() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap forward_signal USR1 TERM
"$PYTHON" -I scripts/v10_epfl/rulsif_v2_timing.py "$@" &
child=$!
set +e
wait "$child"; rc=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; rc=$?; fi
echo "end=$(date -u +%FT%TZ) exit=${rc}"
exit "$rc"
