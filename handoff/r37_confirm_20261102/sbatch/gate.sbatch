#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r37_confirm_20261102/logs/%x_%j.out
# R37 GPU gate (technical only). QOS/partition on the sbatch command line.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r37_confirm_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
$PY -u -B $W/r37/gate.py
