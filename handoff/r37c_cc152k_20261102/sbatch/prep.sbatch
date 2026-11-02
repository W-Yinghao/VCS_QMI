#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r37c_cc152k_20261102/logs/%x_%j.out
# R37C: CC152K parent cache (native DEV reproduction, exact, else no manifest) + prep (locks, calibration, top-32, B-TRUST).
# QOS/partition on the command line. REQ_GPU = GPU name the native DEV reproduction must run on (parent best epoch 38 was
# validated on RTX PRO 6000, R31 attempt 2). Stages train/dev only (no test file).
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r37c_cc152k_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "start $(date -u +%FT%TZ) host $(hostname)"; t0=$(date +%s)
source $W/sbatch/stage_cc.sh
$PY -u -B $W/r37c/build_cache.py --cond cc --data-path "$DP" ${REQ_GPU:+--require-gpu "$REQ_GPU"}
echo "cache done $(( $(date +%s)-t0 ))s"
[ -f /projects/EEG-foundation-model/yinghao/outputs_v9/r37c_cc152k_20261102/prep/PREP_MANIFEST.json ] || $PY -u -B $W/r37c/prep.py --data-path "$DP"
echo "R37C_PREP_JOB_END elapsed_s $(( $(date +%s)-t0 )) $(date -u +%FT%TZ)"
