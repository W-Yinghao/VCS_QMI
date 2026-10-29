#!/bin/bash
#SBATCH --job-name=r27_audit_ranks
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_audit_20261029/logs/%x_%j.out
# Audit 3: per-query full-DEV ranks for every finished R27 unit + six A0 (R25 H0). DEV only. Rerunnable (done units skipped).
set -euo pipefail
D=/home/infres/yinwang/CS_QMI/handoff/r27_audit_20261029
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_audit_20261029/audit3_ranks
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $D/sbatch/stage_dev.sh
cd $D
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -B -u audit_ranks.py $O $DP
echo RANKS_DONE
