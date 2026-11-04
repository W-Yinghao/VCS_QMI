#!/usr/bin/env bash
# usage: sbatch nr1b_readonly.sbatch <run dirs...>
#SBATCH --job-name=nr1b_readonly
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr1_20261104/sbatch/common_env.sh
export NR1_FEATURE_RAM=all
$NR1_PY -B site/nr1b_readonly.py --runs "$@"
