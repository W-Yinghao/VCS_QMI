#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103/logs/%x_%j.out
# R43 B: R37 B-TRUST subsets (native IN2R eval_train, seeds 1001) of the three 80% Flickr parents + their R43 prep dirs (tools/r43_build_trust.py).
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "start $(date -u +%FT%TZ)"; t0=$(date +%s)
source /home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/sbatch/stage.sh
$PY -u -B $W/tools/r43_build_trust.py --data-path "$DP"
echo "R43_TRUST_JOB_END elapsed_s $(( $(date +%s)-t0 )) $(date -u +%FT%TZ)"
