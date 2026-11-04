#!/usr/bin/env bash
# usage: sbatch --job-name=cr1_x cpu_py.sbatch <command...>
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --nice=50
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr1_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr1_20261104/sbatch/common_env.sh
echo "JOB $SLURM_JOB_ID $(hostname) $(date -u +%FT%TZ)"
eval "$@"
echo "EXIT_OK $(date -u +%FT%TZ)"
