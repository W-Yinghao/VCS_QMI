#!/usr/bin/env bash
#SBATCH --job-name=resume_bf1
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/infres/yinwang/CS_QMI/env/bin/python -u /home/infres/yinwang/CS_QMI/handoff/v10_fill_validate_20261019/activate.py
