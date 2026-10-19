#!/usr/bin/env bash
#SBATCH --job-name=v10_geometry_check
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
export V10_VALIDATION_SOURCE=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_pipeline_20261019
export V10_VALIDATION_MANIFEST=/home/infres/yinwang/CS_QMI/handoff/v10_method_pipeline_20261019/validation/MANIFEST.json
export V10_VALIDATION_SHA256=8befdabb0c2b6823e8c04c86763090fd2b383b28b5c02250c34fcae15eb1292b
/home/infres/yinwang/CS_QMI/env/bin/python -u /home/infres/yinwang/CS_QMI/handoff/v10_geometry_audit_20261019/diagnose.py
