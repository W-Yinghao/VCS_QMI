#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=cpu-high,CPU
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r36_supervision_20261102/logs/%x_%j.out
# R36 evaluator-only candidate audit (package SERVER_PROMPT item 1): export -> package tool (python -I -B, by path) -> numbers-only tables
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r36_supervision_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=
echo "host $(hostname) start $(date -u +%FT%TZ)"
$PY -u -B $W/audit/audit_export.py --stage ${STAGE:-all}
echo "R36_AUDIT_JOB_END $(date -u +%FT%TZ)"
