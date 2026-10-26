#!/usr/bin/env bash
#SBATCH --qos=normal
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset PYTHONPATH
cd /home/infres/yinwang/CS_QMI/handoff/r23_mechanism_toy_20261026
exec /home/infres/yinwang/CS_QMI/env/bin/python -B -s mech.py "$@"
