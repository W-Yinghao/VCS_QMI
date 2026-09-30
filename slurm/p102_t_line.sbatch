#!/usr/bin/env bash
#SBATCH --job-name=p102_t_line
#SBATCH --partition=RTX6000PRO,H100,A100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p102_t_line_%j.out
# P102 T line: conditional dependence increments on the T1 task (both encoders, colour + blur, strengths incl. 0; 5 refit repeats; 1000 bootstrap reps).
# Resumable at the cell level is NOT needed (≈ 1–2 h); OUT prefix via the environment (default reports/P102_t_line_results).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
OUT=${OUT:-reports/P102_t_line_results}
banner "P102 T-line increments -> $OUT"
"$PYTHON" scripts/t_line_increments.py --out "$OUT" ${EXTRA_ARGS:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
