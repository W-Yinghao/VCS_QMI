#!/usr/bin/env bash
#SBATCH --job-name=p93_d1_p72_adapters
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p93_d1_%j.out
# P93 D1: P72 adapter drift decomposition (disclosed retraining through robust_r2r3.train_r), both shifts, m in {0, 0.4}, four methods.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
OUT=${OUT:-reports/P94_d1_p72_adapters}
banner "P93 D1 -> $OUT"
"$PYTHON" scripts/diag_p72_adapters.py --p72 reports/P72_robust_r2r3.json --out "$OUT" --ckpt-dir /home/infres/yinwang/CS_QMI/outputs/P93_D_diag/p72_adapters; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
