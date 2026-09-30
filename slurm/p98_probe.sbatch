#!/usr/bin/env bash
#SBATCH --job-name=p98_probe_robustness
#SBATCH --partition=RTX6000PRO,H100,A100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p98_probe_%j.out
# P98: probe-robustness re-evaluation of the 18 P91 CIFAR-100 checkpoints + 9 CIFAR-10 8x reference cells (evaluation only, selection split).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P98 probe robustness"
"$PYTHON" scripts/probe_robustness.py --out reports/P98_probe_robustness ${EXTRA:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
