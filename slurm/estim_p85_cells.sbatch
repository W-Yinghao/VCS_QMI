#!/usr/bin/env bash
#SBATCH --job-name=p85_cells
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p85_%x_%j.out
# P85 estimator benchmark: every cell of UNITS on one GPU (resumable: finished cell JSONs are skipped). UNITS and OUT required; METHODS optional.
#   sbatch --job-name=p85_pilot --export=ALL,UNITS=slurm/p85_units/pilot.txt,OUT=/home/infres/yinwang/CS_QMI/outputs/P85_estim_benchmark slurm/estim_p85_cells.sbatch
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${UNITS:?}"; : "${OUT:?}"
banner "P85 cells: $UNITS -> $OUT (methods ${METHODS:-from units})"
"$PYTHON" scripts/estim_p85.py --units "$UNITS" --out "$OUT" ${METHODS:+--methods $METHODS}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
