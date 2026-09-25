#!/usr/bin/env bash
#SBATCH --job-name=probe_std
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/probe_std_%j.out
# Evaluation-side diagnostic: standardized-feature linear probe on all completed runs' cached features (no training).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "standardized probe diagnostic"
"$PYTHON" scripts/probe_standardized.py --output-root "$OUTPUT_ROOT" --out reports/PROBE_STANDARDIZED.md
rc=$?; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
