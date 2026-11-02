#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=L40S
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r37_confirm_20261102/logs/%x_%j.out
# R37 new-parent caches (R34 build_cache rule, both peers, native DEV reproduction) + prep (R35 rules) for
# Flickr60 H parents s1002/s1003. L40S only: both parents trained on L40S and f60_s1003 HH reproduces its DEV only on L40S.
# Stages train/dev only (no test file).
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r37_confirm_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "start $(date -u +%FT%TZ)"; t0=$(date +%s)
source /home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/sbatch/stage.sh
for C in n60s1002 n60s1003; do $PY -u -B $W/r37/build_cache.py --cond $C --data-path "$DP" --require-gpu "NVIDIA L40S"; done
echo "caches done $(( $(date +%s)-t0 ))s"
[ -f /projects/EEG-foundation-model/yinghao/outputs_v9/r37_confirm_20261102/prep/PREP_MANIFEST.json ] || $PY -u -B $W/r37/prep.py --data-path "$DP" --conds n60s1002,n60s1003
echo "R37_PREP_JOB_END elapsed_s $(( $(date +%s)-t0 )) $(date -u +%FT%TZ)"
