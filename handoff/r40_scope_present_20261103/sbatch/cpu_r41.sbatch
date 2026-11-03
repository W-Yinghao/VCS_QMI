#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=16
#SBATCH --mem=170G
#SBATCH --time=04:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103/logs/%x_%j.out
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=
echo "host $(hostname) start $(date -u +%FT%TZ) R41 CPU"; t0=$(date +%s)
$PY -I -B $W/tests/cpu_test_r41.py ${R41_PART}
echo "R41_CPU_DONE elapsed_s $(( $(date +%s)-t0 ))"
