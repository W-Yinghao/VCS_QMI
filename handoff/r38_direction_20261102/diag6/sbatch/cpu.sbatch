#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/diag6/logs/%x_%j.out
set -uo pipefail
W6=/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/diag6
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1
echo "=== $(date -u +%FT%TZ) $(hostname) $@"
$PY -u -B $W6/"$@"; rc=$?
echo "DIAG6_END rc=$rc $(date -u +%FT%TZ)"
exit $rc
