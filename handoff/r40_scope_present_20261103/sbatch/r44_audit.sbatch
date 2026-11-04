#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103/logs/%x_%j.out
# R44 audit (zero training, no TEST): tools/r44_audit.py <tree> <cond> <items>
set -uo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103; PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name --format=csv
$PY -u -B $W/tools/r44_audit.py "$1" "$2" "$3"
