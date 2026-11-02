#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102/logs/%x_%j.out
# R39 prep: $1 = tree (r39f|r39c), $2 = complete|joint, $3 = smoke unit (joint only). QOS/partition/nice on the command line.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "start $(date -u +%FT%TZ) host $(hostname) tree $1 stage $2"; t0=$(date +%s)
if [ "$2" = complete ]; then $PY -u -I -B $W/tools/prep_complete.py --tree $1
else $PY -u -I -B $W/tools/prep_joint.py --tree $1 --smoke-unit "$3" ${4:-}; fi
echo "R39_PREP_JOB_END elapsed_s $(( $(date +%s)-t0 )) $(date -u +%FT%TZ)"
