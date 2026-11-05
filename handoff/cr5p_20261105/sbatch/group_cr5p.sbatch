#!/usr/bin/env bash
# usage: sbatch --job-name=cr5p_<COND>_t<SEED> group_cr5p.sbatch <COND> <SEED>   (one matched group = one job = one GPU)
#SBATCH --qos=normal
#SBATCH --partition=L40S,RTX6000PRO,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr5p_20261105/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr5p_20261105/sbatch/common_env.sh
echo "JOB $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)"
$CR5P_PY -I -B site/cr5p_group.py --condition $1 --seed $2
echo "EXIT_OK $(date -u +%FT%TZ)"
