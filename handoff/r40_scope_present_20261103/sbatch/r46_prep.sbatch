#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103/logs/%x_%j.out
# R46 prep: $1 = subset (R46-A pools, tools/r46_build_subset.py) | r2cand <tree> <cond> (R46-B round-2 tables, tools/r46_build_r2cand.py).
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; echo "start $(date -u +%FT%TZ) $(hostname)"; t0=$(date +%s)
if [ "$1" = subset ]; then $PY -u -B $W/tools/r46_build_subset.py; else $PY -u -B $W/tools/r46_build_r2cand.py --tree $2 --cond $3; fi
echo "R46_PREP_END elapsed_s $(( $(date +%s)-t0 )) $(date -u +%FT%TZ)"
