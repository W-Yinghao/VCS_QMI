#!/usr/bin/env bash
#SBATCH --job-name=estim_cells
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/estim_%x_%j.out
# E line / R1: run every cell of a units file on one GPU. UNITS and OUT required; EXTRA_ARGS passed through (e.g. "--levels 1" for R1).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${UNITS:?}"; : "${OUT:?}"
banner "estimator benchmark cells: $UNITS -> $OUT (${EXTRA_ARGS:-})"
"$PYTHON" scripts/estim_staircase.py --units "$UNITS" --out "$OUT" ${EXTRA_ARGS:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
