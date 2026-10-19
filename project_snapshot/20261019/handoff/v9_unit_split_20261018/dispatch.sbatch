#!/usr/bin/env bash
#SBATCH --job-name=v9_split_feed
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/infres/yinwang/CS_QMI/handoff/v9_unit_split_20261018
/home/infres/yinwang/CS_QMI/env/bin/python -u transition.py
/home/infres/yinwang/CS_QMI/env/bin/python -u dispatch.py
