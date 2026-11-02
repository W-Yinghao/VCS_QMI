#!/bin/bash
#SBATCH --job-name=r37s_cpu
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102/logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -u -B /home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102/tools/cpu_check.py
