#!/usr/bin/env bash
#SBATCH --job-name=ssl_aggregate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Frozen SSL aggregation scripts given as arguments (main env), e.g. sbatch slurm/ssl_aggregate_cpu.sbatch scripts/p130a1_aggregate.py
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; rc=0
for s in "$@"; do echo "== $s"; "$PYTHON" "$s" || rc=1; done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
