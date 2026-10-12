#!/usr/bin/env bash
#SBATCH --job-name=r6reg3_aggregate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R6 REG-P3 readings (scripts/r6_reg_p3_aggregate.py [--smoke] [--root R] [--out DIR]); main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
"$PYTHON" scripts/r6_reg_p3_aggregate.py "$@"; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
