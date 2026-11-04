#!/usr/bin/env bash
# usage: sbatch --job-name=cr3a1_<COND> group_a1.sbatch <COND>   (CR3_A1: whole matched group on ONE GPU, any allowed type)
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr3_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr3_20261104/sbatch/common_env.sh
echo "JOB $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)"
$CR3_PY -I -B site/cr3_group_a1.py --condition $1
echo "EXIT_OK $(date -u +%FT%TZ)"
