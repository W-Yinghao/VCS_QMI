#!/usr/bin/env bash
#SBATCH --job-name=r4_t2
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R4-T2 fixed-critic sampling study (scripts/r4_t2_sampling.py); args passed through (e.g. --stage 1a [--smoke]); main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
"$PYTHON" scripts/r4_t2_sampling.py --out /home/infres/yinwang/CS_QMI/outputs/R4_T2 --workers 16 "$@"; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
