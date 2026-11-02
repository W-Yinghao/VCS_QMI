#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r34_direct_critic_20261102/logs/%x_%j.out
# R34 cache build for both conditions (stage train/dev only, no test files), one GPU
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r34_direct_critic_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source /home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/sbatch/stage.sh
for C in "$@"; do $PY -u -B $W/r34/build_cache.py --cond $C --data-path "$DP"; done
