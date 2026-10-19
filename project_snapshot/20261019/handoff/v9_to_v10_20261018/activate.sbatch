#!/usr/bin/env bash
#SBATCH --job-name=v10_queue_setup
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 TZ=UTC
cd /home/infres/yinwang/CS_QMI/handoff/v9_to_v10_20261018
/home/infres/yinwang/CS_QMI/env/bin/python -u activate.py
