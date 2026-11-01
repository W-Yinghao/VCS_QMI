#!/bin/bash
#SBATCH --qos=runfill
#SBATCH --partition=V100-32GB,L40S,A100,A40
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --nice=0
#SBATCH --time=10:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r30_diag_20261101/logs/%x_%j.out
# R30 v1.1 probe v3 (DIAG_PROTOCOL_V3.json), read-only: states opened mmap read-only; outputs only under r30_diag_20261101.
# Usage: margin_gpu_v3.sbatch COND SEED REF_STATE REF_ID LABEL=PATH[=UNIT_ID] ...   (env R30DIAG_OUT, R30DIAG_SMOKE=1 optional)
# Per-state output files let a preempted/requeued runfill job continue where it stopped.
set -euo pipefail
COND=$1; SEED=$2; REF=$3; REFID=$4; shift 4
export PYTHONDONTWRITEBYTECODE=1
D=/home/infres/yinwang/CS_QMI/handoff/r30_diag_20261101
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source /home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/sbatch/stage.sh   # train/dev only (no TEST) -> $DP
cd $D
SM=""; [ "${R30DIAG_SMOKE:-0}" = 1 ] && SM="--smoke"
$PY -u -B tools/margin_diag_v3.py --cond $COND --seed $SEED --ref $REF --ref-id $REFID --out-dir ${R30DIAG_OUT:-$D/states_out_v3} --data-path $DP $SM --states "$@"
echo MARGIN_V3_DONE $COND $SEED
