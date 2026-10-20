#!/usr/bin/env bash
#SBATCH --job-name=v10e_r2xhw
#SBATCH --qos=normal
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Cross-hardware RuLSIF_exact_v2 GPU gate (owner 2026-10-19: finals on RTX6000PRO / L40S / H100; CAL stays on L40S): the same gate script
# with --xhw (the GPU must be one of the three accepted final types), every part incl. capacity2048, on the GPU type given by --partition,
# from the frozen source of the v2 releases.  A PASS record is the precondition for any v2 final on that GPU type.
#   V10_EPFL_SOURCE=$S V10_EPFL_EXPECT_COMMIT=$(git -C $S rev-parse HEAD) V10_EPFL_RECORD_SHA256=<GATE_CHECKS record> \
#   V10_EPFL_GROUP_SHA256=<group record> V10_EPFL_OUT=<v2 bundle>/xhw_gates \
#     sbatch --export=ALL --partition=<RTX6000PRO|H100> --job-name=v10e_r2xhw_<label> $S/slurm/v10_epfl/r2_gpu_gate_xhw.sbatch
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
"$PYTHON" -I scripts/v10_epfl/rulsif_v2_gpu_gate.py --xhw "$@" &
child=$!
set +e
wait "$child"; rc=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; rc=$?; fi
echo "end=$(date -u +%FT%TZ) exit=${rc}"
exit "$rc"
