#!/usr/bin/env bash
#SBATCH --job-name=r6reg2_aggregate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R6 REG-P2 readings (scripts/r6_reg_p2_aggregate.py [--smoke]); main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
"$PYTHON" scripts/r6_reg_p2_aggregate.py "$@"; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
