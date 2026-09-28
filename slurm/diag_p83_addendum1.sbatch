#!/usr/bin/env bash
#SBATCH --job-name=p83_addendum1_supplement
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p83_addendum1_%j.out
# P83 addendum 1: P84 supplement (actual gate, dictionary disagreement bounds, converged cosine, ridge member, Jacobians) on the ten P84 checkpoints.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
OUT_DIR=${OUT_DIR:-/home/infres/yinwang/CS_QMI/outputs/P83_addendum1_supplement}; REPORT=${REPORT:-reports/P84_addendum1_supplement.md}
banner "P83 addendum 1 -> $OUT_DIR / $REPORT"
"$PYTHON" scripts/diag_p84_supplement.py --out-dir "$OUT_DIR" --report "$REPORT" --workers 7; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
