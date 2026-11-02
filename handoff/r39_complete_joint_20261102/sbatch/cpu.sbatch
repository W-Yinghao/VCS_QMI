#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=cpu-high,CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102/logs/%x_%j.out
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=
echo "host $(hostname) start $(date -u +%FT%TZ) part $1"; t0=$(date +%s)
$PY -I -B $W/tests/cpu_test.py $1
[ "$1" = post ] && $PY -I -B $W/tools/audit_completion_f60.py
echo "R39_CPU_DONE elapsed_s $(( $(date +%s)-t0 ))"
