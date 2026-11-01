#!/bin/bash
#SBATCH --job-name=r30diag_queue
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r30_diag_20261101/logs/%x_%j.out
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
D=/home/infres/yinwang/CS_QMI/handoff/r30_diag_20261101
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -B $D/tools/export_queue_r29.py /projects/EEG-foundation-model/yinghao/outputs_v9/r29_listwise_20261101/runs $D/R29_QUEUE_SAVED_STATES.json
