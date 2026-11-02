#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=cpu-high,CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102/logs/%x_%j.out
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
echo "host $(hostname) start $(date -u +%FT%TZ) part $1"; t0=$(date +%s)
$PY -I -B $W/tests/cpu_test.py $1
echo "R38_CPU_DONE elapsed_s $(( $(date +%s)-t0 ))"
