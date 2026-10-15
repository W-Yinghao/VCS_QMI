#!/usr/bin/env bash
#SBATCH --job-name=v9_reg_g2_aggregate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-G2 aggregate (CPU) after the 10 grouped jobs: unit-pooled readings, contrasts, primary family (Holm over 4), interaction -> reports/V9/V9_G2_results/.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v9; cd "$R" || exit 2; export PYTHONPATH="$R/src:$R"
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git -C "$R" rev-parse --short HEAD)"
"$PYTHON" scripts/v9/reg_g2.py aggregate --out "$R/reports/V9/V9_G2_results"; rc=$?
echo "aggregate_rc=$rc end=$(date -u +%FT%TZ)"; exit $rc
