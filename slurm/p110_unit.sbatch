#!/usr/bin/env bash
#SBATCH --job-name=p110_unit
#SBATCH --partition=RTX6000PRO,H100,A100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P110 bundle job (package v4 V1 / V2 / V3): runs MODULES for the encoders in RUNS sequentially in ONE GPU job; finished result files are skipped.
# Env: RUNS (comma list), MODULES (default v1,v2,v3), OUT (default reports/P110), EXTRA (e.g. --smoke).  Normal QOS only.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${RUNS:?}"
banner "P110 bundle: modules=${MODULES:-v1,v2,v3} runs=$RUNS out=${OUT:-reports/P110} ${EXTRA:-}"
"$PYTHON" scripts/p110_v.py --runs "$RUNS" --modules "${MODULES:-v1,v2,v3}" --out "${OUT:-reports/P110}" ${EXTRA:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
