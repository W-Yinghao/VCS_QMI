#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103/logs/%x_%j.out
# R42: cache + prep + JOINT prep of one IN2R-native Flickr parent ($1 = f40in|f80in), on the parent's training GPU type ($2).
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "start $(date -u +%FT%TZ)"; t0=$(date +%s)
source /home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/sbatch/stage.sh
$PY -u -B $W/tools/r42_build_parent.py --cond $1 --data-path "$DP" --require-gpu "$2"
echo "R42_PARENT_JOB_END elapsed_s $(( $(date +%s)-t0 )) $(date -u +%FT%TZ)"
