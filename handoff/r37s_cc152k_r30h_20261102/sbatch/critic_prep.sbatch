#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102/logs/%x_%j.out
# R37S critic line: cache of the R37S parent (exact native DEV reproduction required, else no manifest; --require-gpu = GPU type of the
# parent's best epoch) + prep (locks, calibration, top-32, B-TRUST) -- R38 prep.sbatch rule. QOS/partition on the command line.
# Stages train/dev only (sbatch/stage_cc.sh of this line; no test file). $1 = cond (cc_r37s), $2 = required GPU name.
set -euo pipefail
COND=$1; REQ=$2
W=/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "start $(date -u +%FT%TZ) host $(hostname) cond $COND"; t0=$(date +%s)
source $W/sbatch/stage_cc.sh
$PY -u -B $W/r37s_critic/build_cache.py --cond $COND --data-path "$DP" --require-gpu "$REQ"
echo "cache done $(( $(date +%s)-t0 ))s"
[ -f /projects/EEG-foundation-model/yinghao/outputs_v9/r37s_cc152k_r30h_20261102/critic/prep/$COND/PREP_MANIFEST.json ] || $PY -u -B $W/r37s_critic/prep.py --cond $COND --data-path "$DP"
echo "R37S_PREP_JOB_END elapsed_s $(( $(date +%s)-t0 )) $(date -u +%FT%TZ)"
