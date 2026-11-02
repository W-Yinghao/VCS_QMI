#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r35_expanded_20261102/logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= CUBLAS_WORKSPACE_CONFIG=:4096:8
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -u -B /home/infres/yinwang/CS_QMI/handoff/r35_expanded_20261102/tests/cpu_sim.py
