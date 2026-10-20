#!/usr/bin/env bash
#SBATCH --job-name=v10e_xhwgate
#SBATCH --qos=normal
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Cross-hardware EPFL item-5 GPU gate (owner 2026-10-19 / 2026-10-20): scripts/v10_epfl/gpu_gate_xhw.py = gpu_gate.py, every part and every
# family's required row, with the two checkpoint energy / gradient rows on T01's round-trip tolerance (state hash bitwise), on the GPU type
# given by --partition (RTX6000PRO or H100; L40S refused: its gate is 1048825), from the frozen hardware-release source.  Stage
# GATE_CHECKS (FIT / CAL only).  A PASS record under <hw bundle>/xhw_gates/<label>/ is the precondition for any final on that GPU type.
#   S=<frozen hw source>; B=<hw bundle>
#   V10_EPFL_SOURCE=$S V10_EPFL_EXPECT_COMMIT=$(git -C $S rev-parse HEAD) \
#   V10_EPFL_RECORD_SHA256=$(sha256sum $S/v10/epfl/EPFL_ACTIVATION_RECORD.json | cut -c1-64) V10_EPFL_OUT=$B/xhw_gates/<label> \
#     sbatch --export=ALL --partition=<RTX6000PRO|H100> --job-name=v10e_xhwgate_<label> $S/slurm/v10_epfl/gpu_gate_xhw.sbatch
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
: "${SLURM_JOB_ID:?Slurm required}"
: "${V10_EPFL_SOURCE:?frozen hardware-release source required}"
: "${V10_EPFL_EXPECT_COMMIT:?commit pin required}"
: "${V10_EPFL_RECORD_SHA256:?GATE_CHECKS activation record pin required}"
: "${V10_EPFL_OUT:?output root <hw bundle>/xhw_gates/<label> required}"
R=$V10_EPFL_SOURCE
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
mkdir -p "$V10_EPFL_OUT"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) job=${SLURM_JOB_ID} partition=${SLURM_JOB_PARTITION:-NA} commit=$(git -C "$R" rev-parse HEAD) expect=${V10_EPFL_EXPECT_COMMIT} record_pin=${V10_EPFL_RECORD_SHA256} out=${V10_EPFL_OUT}"
echo "precision_env CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG-<unset>} TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE-<unset>} NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE-<unset>}"
cd "$R"
child=''
forward_signal() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap forward_signal USR1 TERM
"$PYTHON" -I scripts/v10_epfl/gpu_gate_xhw.py "$@" &
child=$!
set +e
wait "$child"; rc=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; rc=$?; fi
echo "end=$(date -u +%FT%TZ) exit=${rc}"
exit "$rc"
