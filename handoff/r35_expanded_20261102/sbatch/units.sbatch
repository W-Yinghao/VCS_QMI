#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r35_expanded_20261102/logs/%x_%j.out
# R35 formal: each argument "runner:unit1,unit2,..." runs as its own python process, in order, on this GPU
# (runner = train_unit | train_oracle). DONE units are skipped; checkpoints are resumed. A failed process does
# not stop the later ones (each unit is a separate technical failure record); exit code = any failure.
set -uo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r35_expanded_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
rc=0
for spec in "$@"; do
  runner=${spec%%:*}; units=${spec#*:}
  echo "=== $(date -u +%FT%TZ) $runner $units"
  $PY -u -B $W/r35/$runner.py --units "$units" || rc=1
done
echo "R35_JOB_END rc=$rc $(date -u +%FT%TZ)"
exit $rc
