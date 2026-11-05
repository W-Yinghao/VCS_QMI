#!/usr/bin/env bash
# usage: sbatch --job-name=cr4_x gpu_py.sbatch <command...>
#SBATCH --partition=L40S,RTX6000PRO,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --nice=50
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr4_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr4_20261104/sbatch/common_env.sh
DEV=$( command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
echo "JOB $SLURM_JOB_ID $(hostname) ${DEV:+$DEV }$(date -u +%FT%TZ)"
eval "$@"
echo "EXIT_OK $(date -u +%FT%TZ)"
