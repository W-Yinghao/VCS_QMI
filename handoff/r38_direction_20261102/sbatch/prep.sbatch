#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/logs/%x_%j.out
# R38: CC152K new-parent cache (native DEV reproduction, exact, else no manifest) + prep (locks, calibration, top-32, B-TRUST)
# for one parent ($1 = cc_s1002 | cc_s1003). R37C prep.sbatch rule. QOS/partition on the command line. Stages train/dev only (no test file).
set -euo pipefail
COND=$1
W=/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "start $(date -u +%FT%TZ) host $(hostname) cond $COND"; t0=$(date +%s)
source $W/sbatch/stage_cc.sh
$PY -u -B $W/r38c/build_cache.py --cond $COND --data-path "$DP"
echo "cache done $(( $(date +%s)-t0 ))s"
[ -f /projects/EEG-foundation-model/yinghao/outputs_v9/r38_direction_20261102/prep/$COND/PREP_MANIFEST.json ] || $PY -u -B $W/r38c/prep.py --cond $COND --data-path "$DP"
echo "R38_PREP_JOB_END elapsed_s $(( $(date +%s)-t0 )) $(date -u +%FT%TZ)"
