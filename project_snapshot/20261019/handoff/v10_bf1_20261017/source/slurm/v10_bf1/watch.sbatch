#!/usr/bin/env bash
#SBATCH --job-name=v10bf_watch
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=3-01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1 TZ=UTC
cd /home/infres/yinwang/CS_QMI/handoff/v10_bf1_20261017/source
python scripts/v10_bf1/watch.py
