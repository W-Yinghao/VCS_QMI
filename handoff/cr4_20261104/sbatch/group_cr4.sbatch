#!/usr/bin/env bash
# usage: sbatch --job-name=cr4_<COND> group_cr4.sbatch <COND>  (one matched group = one job = one GPU)
#SBATCH --qos=normal
#SBATCH --partition=L40S,RTX6000PRO,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr4_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr4_20261104/sbatch/common_env.sh
echo "JOB $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)"
$CR4_PY -I -B site/cr4_group.py --condition $1
echo "EXIT_OK $(date -u +%FT%TZ)"
