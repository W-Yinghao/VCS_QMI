#!/usr/bin/env bash
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr4_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr4_20261104/sbatch/common_env.sh
$NR1_PY -B "$@"
