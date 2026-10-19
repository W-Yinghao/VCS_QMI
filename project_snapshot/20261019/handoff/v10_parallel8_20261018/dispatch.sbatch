#!/usr/bin/env bash
#SBATCH --job-name=dispatch_bf1
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 TZ=UTC OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /home/infres/yinwang/CS_QMI/handoff/v10_parallel8_20261018
/home/infres/yinwang/CS_QMI/env/bin/python -u dispatch.py "$@"
