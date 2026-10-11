#!/usr/bin/env bash
#SBATCH --job-name=r5_qnorm_aggregate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R5-QN frozen readings (scripts/r5_qnorm_aggregate.py); main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
"$PYTHON" scripts/r5_qnorm_aggregate.py; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
