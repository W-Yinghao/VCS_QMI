#!/usr/bin/env bash
#SBATCH --job-name=v10bf_preflight
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
/home/infres/yinwang/CS_QMI/env/bin/python /home/infres/yinwang/CS_QMI/ssl_pilot_v10_bf1/scripts/v10_bf1/preflight.py
