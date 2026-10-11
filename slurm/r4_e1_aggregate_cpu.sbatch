#!/usr/bin/env bash
#SBATCH --job-name=r4_e1_aggregate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R4-E1 frozen readings (scripts/r4_e1_aggregate.py; args passed through, e.g. --N 16384 for the separate extension); main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail
"$PYTHON" scripts/r4_e1_aggregate.py "$@"; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
