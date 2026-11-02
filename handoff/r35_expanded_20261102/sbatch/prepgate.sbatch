#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r35_expanded_20261102/logs/%x_%j.out
# R35 prep (once: locks, common calibration, FIT candidate tables, tokens, B-TRUST native GMM) then GPU gate
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r35_expanded_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if [ ! -f /projects/EEG-foundation-model/yinghao/outputs_v9/r35_expanded_20261102/prep/PREP_MANIFEST.json ]; then
  source /home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/sbatch/stage.sh
  $PY -u -B $W/r35/prep.py --data-path "$DP"
fi
[ "${1:-}" = "prep-only" ] && exit 0
$PY -u -B $W/r35/gate.py
