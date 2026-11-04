#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103/logs/%x_%j.out
# R43 A: tokens (c60in_s1001 only) + cache + prep + JOINT prep of one IN2R-native R33 COCO parent ($1) on its training GPU type ($2)
# (tools/r43_build_coco.py). $3 = 1: afterwards a 30-update timing run of the frozen J and joint J configs of this parent in the trainer's
# gate mode (scratch dir r43_parents/timing/<cond>; not a formal unit, never reused) to register COCO memory / time.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "start $(date -u +%FT%TZ) $(hostname)"; t0=$(date +%s)
DATA_NAME=coco_precomp; source /home/infres/yinwang/CS_QMI/handoff/r33_baselines_20261102/sbatch/stage_ds.sh   # sets DP (train/dev only, never TEST)
$PY -u -B $W/tools/r43_build_coco.py --cond $1 --data-path "$DP" --require-gpu "$2"
echo "R43_BUILD_END elapsed_s $(( $(date +%s)-t0 )) $(date -u +%FT%TZ)"
if [ "${3:-0}" = 1 ]; then
  P=$(echo $1 | sed -E 's/^c([0-9]+)in_s(100[0-9])$/C\1_IN2R_H\2/')
  for K in frozen joint; do
    t1=$(date +%s)
    /usr/bin/time -v $PY -u -B $W/r43o/train_unit.py --units R43_A_${P}_N_${K}_J --gate-out /projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103/r43_parents/timing/$1 --gate-steps 30 2>&1 | grep -E "R41_UNIT_DONE|R41_JOB_END|Maximum resident|Elapsed|Error|Traceback|assert" || true
    echo "R43_TIMING $K elapsed_s $(( $(date +%s)-t1 ))"
  done
fi
echo "R43_PARENT_JOB_END elapsed_s $(( $(date +%s)-t0 )) $(date -u +%FT%TZ)"
