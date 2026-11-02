#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r34_direct_critic_20261102/logs/%x_%j.out
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r34_direct_critic_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
$PY -u -B $W/r34/gate.py
