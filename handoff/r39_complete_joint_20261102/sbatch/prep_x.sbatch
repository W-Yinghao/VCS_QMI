#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102/logs/%x_%j.out
# R39X JOINT prep for one additional parent: $1 = tree (r39xf|r39xc), $2 = cond. QOS/partition/nice/mem on the command line.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "start $(date -u +%FT%TZ) host $(hostname) tree $1 cond $2"; t0=$(date +%s)
$PY -u -I -B $W/tools/prep_joint_x.py --tree $1 --cond $2
echo "R39X_PREP_JOB_END elapsed_s $(( $(date +%s)-t0 )) $(date -u +%FT%TZ)"
