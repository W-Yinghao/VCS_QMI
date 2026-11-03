#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103/logs/%x_%j.out
# R40 section 7 zero-training readouts: $1 = tree dir, $2 = tag, $3 = unit dirs (comma), $4 = common J unit dir (item 3) or ''
set -uo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if [ -n "${4:-}" ]; then $PY -u -B $W/diag7/readouts.py "$1" "$2" "$3" --common "$4"; else $PY -u -B $W/diag7/readouts.py "$1" "$2" "$3"; fi
rc=$?
if [ $rc -eq 0 ] && [[ "$2" == f60_* ]]; then $PY -u -I -B $W/diag7/eval_truth_f60.py "$2"; rc=$?; fi
echo "R40_DIAG7_END rc=$rc"; exit $rc
