#!/usr/bin/env bash
# usage (driver sets partition/mem/time): sbatch --job-name=cr6_<group> group.sbatch A1 | A2 --seed S | B --condition C
#SBATCH --qos=normal
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr6_20261105/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr6_20261105/sbatch/common_env.sh
echo "JOB $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)"
if [ "$1" = B ]; then export CR6_HIDDEN_RAM=all; fi
$CR6_PY -I -B cr6_group.py "$@"
echo "EXIT_OK $(date -u +%FT%TZ)"
