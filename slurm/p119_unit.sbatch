#!/usr/bin/env bash
#SBATCH --job-name=p119_unit
#SBATCH --partition=RTX6000PRO,H100,A100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P119 bundle job (v5 NEXT-V-PROBE): runs MODULES for the encoders in RUNS sequentially in ONE GPU job; finished result files are skipped.
# Env: RUNS (comma list; default all 15), MODULES (default fpcheck,transfer,c10extra), OUT (default reports/P119), EXTRA (e.g. --fp32 / --smoke).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P119 unit: modules=${MODULES:-fpcheck,transfer,c10extra} out=${OUT:-reports/P119} extra=${EXTRA:-} runs=${RUNS:-default}"
"$PYTHON" scripts/p119_v.py ${RUNS:+--runs "$RUNS"} --modules "${MODULES:-fpcheck,transfer,c10extra}" --out "${OUT:-reports/P119}" ${EXTRA:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
