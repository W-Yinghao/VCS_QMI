#!/usr/bin/env bash
#SBATCH --job-name=v9f_feed
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
cd /home/infres/yinwang/CS_QMI/handoff/v9_full_20261017
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/home/infres/yinwang/CS_QMI/env/bin/python -u feed.py
