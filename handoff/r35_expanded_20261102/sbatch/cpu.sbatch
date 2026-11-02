#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r35_expanded_20261102/logs/%x_%j.out
# R35 CPU stage: resolve matrix/configs, package validate_matrix, CPU test (reference checks + production alignment)
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r35_expanded_20261102
PKG=/home/infres/yinwang/CS_QMI/r35_expanded/VCS_R35_Expanded_Server_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=
echo "host $(hostname) start $(date -u +%FT%TZ)"; t0=$(date +%s)
$PY -B $W/r35/make_configs.py $PKG/configs/MATRIX.csv
$PY -I -B $PKG/tools/validate_matrix.py
$PY -I -B $W/tests/cpu_test.py $PKG
echo "R35_CPU_DONE elapsed_s $(( $(date +%s)-t0 ))"
