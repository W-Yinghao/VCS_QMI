#!/usr/bin/env bash
# usage: sbatch --job-name=cr6_<x> cpu_py.sbatch <script.py> [args]
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --nice=50
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr6_20261105/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr6_20261105/sbatch/common_env.sh
echo "JOB $SLURM_JOB_ID $(hostname) $(date -u +%FT%TZ)"
$CR6_PY -I -B "$@"
echo "EXIT_OK $(date -u +%FT%TZ)"
