#!/bin/bash
#SBATCH --job-name=r27s_cpu
#SBATCH --qos=normal
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/logs/%x_%j.out
set -euo pipefail
D=/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029
cd $D/code_v12
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -u r27s_cpu_check.py $D/S_CPU_CHECK.json
