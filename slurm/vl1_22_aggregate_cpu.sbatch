#!/usr/bin/env bash
#SBATCH --job-name=vl1_22_aggregate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL1-22 Table C readings; main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; "$PYTHON" scripts/vl1_22_aggregate.py; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
