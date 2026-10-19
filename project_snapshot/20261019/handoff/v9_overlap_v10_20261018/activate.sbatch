#!/usr/bin/env bash
#SBATCH --job-name=v10_overlap_setup
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 TZ=UTC
cd /home/infres/yinwang/CS_QMI/handoff/v9_overlap_v10_20261018
/home/infres/yinwang/CS_QMI/env/bin/python -u activate.py
