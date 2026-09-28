#!/usr/bin/env bash
#SBATCH --job-name=estim_pkg_gauss_probe_P81
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/estim_pkg_gauss_probe_%j.out
# Estimator package v1, P81: Gaussian probe (d = 20, I in {0.5, 4, 8}, seed 0).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "estimator probe P81"
rc=0
for I in ${IS:-0.5 4 8}; do
  PYTHONPATH=src "$PYTHON" -m vcs_estim.run --I $I --d 20 --seed 0 --out reports/P82_estim_probe_I${I}_d20_s0.json || rc=1
done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
