#!/usr/bin/env bash
#SBATCH --job-name=r6gain_audit_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r6 GAIN-P1 AV-MNIST audio audit (exact byte duplicates over the train release, twin groups at cos > 0.99, label purity).  Main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
"$PYTHON" scripts/r6_gain_pilot.py audit --dataset avmnist; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
