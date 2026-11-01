#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101/logs/%x_%j.out
# NRCS-v1 generic CPU step: sbatch cpu_cmd.sbatch <command...>   (command run with PYTHONPATH=package/src, python -B)
set -euo pipefail
export W=/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101
export PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=$W/package/src:$W/nrcs:$W/upstream
echo "host $(hostname) start $(date -u +%FT%TZ)"; t0=$(date +%s)
"$@"
echo "elapsed_s $(( $(date +%s)-t0 )) end $(date -u +%FT%TZ)"
