#!/usr/bin/env bash
#SBATCH --job-name=v10_methods
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
: "${V10_PIPELINE_FREEZE:?pinned dispatch freeze required}"
/home/infres/yinwang/CS_QMI/env/bin/python -u /home/infres/yinwang/CS_QMI/handoff/v10_method_pipeline_20261019/dispatch.py
