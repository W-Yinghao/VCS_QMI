#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102/logs/%x_%j.out
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name --format=csv
$PY -u -B $W/tools/replay_update1.py --tree $1 --unit $2
