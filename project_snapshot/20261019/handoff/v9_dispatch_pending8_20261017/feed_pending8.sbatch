#!/usr/bin/env bash
#SBATCH --job-name=v9f_feed
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/infres/yinwang/CS_QMI/handoff/v9_dispatch_pending8_20261017
/home/infres/yinwang/CS_QMI/env/bin/python -u feed_pending8.py
