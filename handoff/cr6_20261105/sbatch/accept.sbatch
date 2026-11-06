#!/usr/bin/env bash
# CR6 real acceptance. Default (L40S, history mode): also bitwise vs the CR1/CR5 L40S history. Other GPUs: sbatch --partition=X accept.sbatch --mode portable [--no-tail]
#SBATCH --qos=normal
#SBATCH --partition=L40S
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr6_20261105/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr6_20261105/sbatch/common_env.sh
echo "JOB $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)"
$CR6_PY -I -B acceptance_cr6.py --work $CR6_OUT/acceptance/acc_$SLURM_JOB_ID "$@"
echo "EXIT_OK $(date -u +%FT%TZ)"
