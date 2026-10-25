#!/usr/bin/env bash
#SBATCH --job-name=r22_repeval_cpu
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/usr/bin/python3 -B /home/infres/yinwang/CS_QMI/handoff/r22_reg_repeat_20261025/static_check.py
/home/infres/yinwang/CS_QMI/env/bin/python -B -s /home/infres/yinwang/CS_QMI/handoff/r22_reg_repeat_20261025/repeat.py check
/usr/bin/python3 /home/infres/yinwang/CS_QMI/handoff/r22_reg_repeat_20261025/admit.py
