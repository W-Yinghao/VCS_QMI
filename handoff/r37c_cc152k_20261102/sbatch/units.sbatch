#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r37c_cc152k_20261102/logs/%x_%j.out
# R37C formal: one python process for the listed units (CC152K parent, same route, 3 objectives), in order on this GPU.
# DONE units are skipped; checkpoints are resumed.
set -uo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r37c_cc152k_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "=== $(date -u +%FT%TZ) $1"
$PY -u -B $W/r37c/train_unit.py --units "$1"; rc=$?
echo "R37C_JOB_END rc=$rc $(date -u +%FT%TZ)"
exit $rc
