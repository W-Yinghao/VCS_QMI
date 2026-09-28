#!/usr/bin/env bash
#SBATCH --job-name=p93_d2_same_class_neg
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p93_d2_%j.out
# P93 D2: same-class negative-pair gradient share on the P84 VCS / SimCLR checkpoints (initial, 100, 400, 800), 16 x 256 FIT images.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
OUT=${OUT:-reports/P94_d2_same_class_negatives}
banner "P93 D2 -> $OUT"
"$PYTHON" scripts/diag_same_class_negatives.py --out "$OUT" ${EXTRA_ARGS:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
