#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/diag6/logs/%x_%j.out
# diag6 GPU job: $@ = script + args (zero-training readouts; no TEST file is opened)
set -uo pipefail
W6=/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/diag6
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "=== $(date -u +%FT%TZ) $(hostname) $@"
$PY -u -B $W6/"$@"; rc=$?
echo "DIAG6_END rc=$rc $(date -u +%FT%TZ)"
exit $rc
