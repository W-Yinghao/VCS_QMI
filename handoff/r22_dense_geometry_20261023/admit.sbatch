#!/usr/bin/env bash
#SBATCH --job-name=r22_dense_admit
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
exec /home/infres/yinwang/CS_QMI/env/bin/python -B /home/infres/yinwang/CS_QMI/handoff/r22_dense_geometry_20261023/admit.py
