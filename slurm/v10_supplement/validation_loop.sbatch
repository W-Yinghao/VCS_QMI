#!/usr/bin/env bash
#SBATCH --job-name=validate_v10
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
: "${V10_VALIDATION_SOURCE:?immutable source required}"
: "${V10_VALIDATION_MANIFEST:?validation manifest required}"
: "${V10_VALIDATION_SHA256:?submission pin required}"
export PYTHONPATH="$V10_VALIDATION_SOURCE/src:$V10_VALIDATION_SOURCE"
cd "$V10_VALIDATION_SOURCE"
/home/infres/yinwang/CS_QMI/env/bin/python -u scripts/v10_supplement/validation_loop.py
