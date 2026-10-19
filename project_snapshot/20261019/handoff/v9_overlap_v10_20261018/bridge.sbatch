#!/usr/bin/env bash
#SBATCH --job-name=audit_v9_end
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TZ=UTC
cd /home/infres/yinwang/CS_QMI/handoff/v9_overlap_v10_20261018
/home/infres/yinwang/CS_QMI/env/bin/python -u bridge.py
