#!/usr/bin/env bash
#SBATCH --job-name=v10e_gate
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --exclude=node51,node60
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V10 EPFL item-5 GPU gate (decision 2026-10-17 section 6.2 item 5): data gate, warp, geometry gradient, pixel units, effective support,
# frozen checkpoint, and the per-family execution gate of the 11-method set on EPFL FIT / CAL windows (stage GATE_CHECKS; DEV refused).
# Prepared by the EPFL activation fork; submitted by the main session after review.  Submit from the clean, committed tree:
#   cd /home/infres/yinwang/CS_QMI/ssl_pilot_v10_epfl
#   V10_EPFL_EXPECT_COMMIT=$(git rev-parse HEAD) V10_EPFL_RECORD_SHA256=$(sha256sum v10/epfl/EPFL_ACTIVATION_RECORD.json | cut -c1-64) \
#     sbatch --export=ALL slurm/v10_epfl/gpu_gate.sbatch
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
: "${SLURM_JOB_ID:?Slurm required}"
: "${V10_EPFL_EXPECT_COMMIT:?commit pin required}"
: "${V10_EPFL_RECORD_SHA256:?activation record pin required}"
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_epfl
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) job=${SLURM_JOB_ID} partition=${SLURM_JOB_PARTITION:-NA} commit=$(git -C "$R" rev-parse HEAD) expect=${V10_EPFL_EXPECT_COMMIT} record_pin=${V10_EPFL_RECORD_SHA256}"
echo "precision_env CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG-<unset>} TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE-<unset>} NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE-<unset>}"
cd "$R"
child=''
forward_signal() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap forward_signal USR1 TERM
"$PYTHON" -I scripts/v10_epfl/gpu_gate.py "$@" &
child=$!
set +e
wait "$child"; rc=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; rc=$?; fi
echo "end=$(date -u +%FT%TZ) exit=${rc}"
exit "$rc"
