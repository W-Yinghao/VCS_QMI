#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=cpu-high,CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102/logs/%x_%j.out
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
echo "host $(hostname) start $(date -u +%FT%TZ)"; t0=$(date +%s)
$PY -I -B $W/tests/critic_cpu_test.py
echo "R37S_CRITIC_CPU_DONE elapsed_s $(( $(date +%s)-t0 ))"
