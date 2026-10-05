#!/usr/bin/env bash
#SBATCH --job-name=p124_unit
#SBATCH --partition=A100,L40S,RTX6000PRO,H100
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P124 bundle job (v6 V6-GRANULARITY): all encoders in RUNS sequentially in ONE GPU job; finished result files are skipped (resumable).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P124 unit: out=${OUT:-reports/P124} runs=${RUNS:-default 12}"
"$PYTHON" scripts/p124_gran.py ${RUNS:+--runs "$RUNS"} --out "${OUT:-reports/P124}" ${EXTRA:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
