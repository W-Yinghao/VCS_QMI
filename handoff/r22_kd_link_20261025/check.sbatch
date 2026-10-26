#!/usr/bin/env bash
#SBATCH --job-name=r22_kdlink_cpu
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /home/infres/yinwang/CS_QMI/env_kd/bin/python -B -s /home/infres/yinwang/CS_QMI/handoff/r22_kd_link_20261025/run.py check
