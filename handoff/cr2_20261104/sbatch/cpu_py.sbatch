#!/usr/bin/env bash
# usage: sbatch --job-name=cr2_x cpu_py.sbatch <command...>
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --nice=50
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr2_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr2_20261104/sbatch/common_env.sh
DEV=$( command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
echo "JOB $SLURM_JOB_ID $(hostname) ${DEV:+$DEV }$(date -u +%FT%TZ)"
eval "$@"
echo "EXIT_OK $(date -u +%FT%TZ)"
