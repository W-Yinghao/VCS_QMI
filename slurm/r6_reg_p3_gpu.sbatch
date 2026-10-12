#!/usr/bin/env bash
#SBATCH --job-name=r6_reg_p3_gpu
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R6 REG-P3 GPU steps (scripts/r6_reg_p3.py train | comir | register ...; --objective / --method / --seed may be comma lists, run in sequence);
# args passed through.  Main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
echo "host=$(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ)"
"$PYTHON" scripts/r6_reg_p3.py "$@"; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
