#!/usr/bin/env bash
#SBATCH --qos=normal
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr6_20261105/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr6_20261105/sbatch/common_env.sh
echo "JOB $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)"
$CR6_PY -I -B "$@"
echo "EXIT_OK $(date -u +%FT%TZ)"
