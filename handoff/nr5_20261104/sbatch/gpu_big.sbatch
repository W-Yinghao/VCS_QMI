#!/usr/bin/env bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr5_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr5_20261104/sbatch/common_env.sh
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
export NR4_HIDDEN_RAM=all NR1_FEATURE_RAM=global
$NR1_PY -B "$@"
