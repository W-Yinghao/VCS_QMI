#!/usr/bin/env bash
#SBATCH --job-name=p141_unit
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P141 bundle job (v7 V7-C100-LAYER-READOUT): all encoders in RUNS sequentially in ONE GPU job; finished result files are skipped (resumable);
# encoders not yet COMPLETED are listed and skipped (resubmit later to pick them up).  RUNS / READOUTS come from the ENVIRONMENT (never put a
# comma-separated value inside --export: SLURM splits on commas) — e.g.  RUNS=a,b,c sbatch --export=ALL slurm/p141_unit.sbatch
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P141 unit: out=${OUT:-reports/P141} readouts=${READOUTS:-recipe_raw,raw_std,knn} runs=${RUNS:-default}"
"$PYTHON" scripts/p141_layers.py --out "${OUT:-reports/P141}" ${EXTRA:-}; rc=$?
"$PYTHON" scripts/p141_aggregate.py --in "${OUT:-reports/P141}" --out reports/P141_v7_layer_readout_results || true
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
