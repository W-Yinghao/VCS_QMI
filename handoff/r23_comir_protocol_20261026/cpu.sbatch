#!/usr/bin/env bash
#SBATCH --qos=normal
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
cd /home/infres/yinwang/CS_QMI/handoff/r23_comir_protocol_20261026
exec /home/infres/yinwang/CS_QMI/env/bin/python -B -s run.py "$@"
