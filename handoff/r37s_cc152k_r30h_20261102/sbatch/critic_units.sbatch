#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102/logs/%x_%j.out
# R37S critic formal (copy of R38 units.sbatch): one python process ($1 = code tree r37s_critic) for the listed units ($2, same parent), in order on this GPU.
# DONE units are skipped; checkpoints are resumed. In-unit fail-fast checks (no standalone gate).
set -uo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "=== $(date -u +%FT%TZ) $(hostname) tree $1 units $2"
$PY -u -B $W/$1/train_unit.py --units "$2"; rc=$?
echo "R37S_JOB_END rc=$rc $(date -u +%FT%TZ)"
exit $rc
