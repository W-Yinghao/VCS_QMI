#!/usr/bin/env bash
# usage: sbatch --job-name=nr2c_x cpu_py.sbatch script.py [args]
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr2_20261104/sbatch/common_env.sh
$NR1_PY -B "$@"
