#!/usr/bin/env bash
#SBATCH --job-name=v10e_xhwdiag
#SBATCH --qos=normal
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Cross-hardware DIAGNOSTIC of the item-5 checkpoint round trip (scripts/v10_epfl/xhw_diag.py): numbers only, never a gate record; it
# changes no release, gate verdict or rule.  FIT data only (the gate's EPFL FIT window and the BF1 / T01 FIT core).
#   S=<frozen diagnostic tree>; B=/home/infres/yinwang/CS_QMI/handoff/v10_epfl_production_hw_20261019
#   V10_EPFL_SOURCE=$S V10_EPFL_EXPECT_COMMIT=$(git -C $S rev-parse HEAD) \
#   V10_EPFL_RECORD_SHA256=$(sha256sum $S/v10/epfl/EPFL_ACTIVATION_RECORD.json | cut -c1-64) V10_DIAG_OUT=$B/diagnostics/xhw_<label> \
#     sbatch --export=ALL --partition=<H100|RTX6000PRO|L40S> --job-name=v10e_xhwdiag_<label> $S/slurm/v10_epfl/xhw_diag.sbatch
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
: "${SLURM_JOB_ID:?Slurm required}"
: "${V10_EPFL_SOURCE:?frozen diagnostic tree required}"
: "${V10_EPFL_EXPECT_COMMIT:?commit pin required}"
: "${V10_EPFL_RECORD_SHA256:?GATE_CHECKS activation record pin required}"
: "${V10_DIAG_OUT:?output root required}"
export V10_EPFL_OUT=$V10_DIAG_OUT
R=$V10_EPFL_SOURCE
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) job=${SLURM_JOB_ID} partition=${SLURM_JOB_PARTITION:-NA} commit=$(git -C "$R" rev-parse HEAD) expect=${V10_EPFL_EXPECT_COMMIT}"
cd "$R"
set +e
"$PYTHON" -I scripts/v10_epfl/xhw_diag.py --out "$V10_DIAG_OUT/$SLURM_JOB_ID"; rc=$?
echo "end=$(date -u +%FT%TZ) exit=${rc}"
exit "$rc"
