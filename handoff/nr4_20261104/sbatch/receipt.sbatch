#!/usr/bin/env bash
# usage: sbatch --partition=<GPU type of the NR3-B references> receipt.sbatch <units...>
#SBATCH --qos=normal
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr4_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr4_20261104/sbatch/common_env.sh
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
export NR1_FEATURE_RAM=global
$NR1_PY -B -m nr4.frozen_receipt "$@"
