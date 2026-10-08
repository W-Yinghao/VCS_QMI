#!/usr/bin/env bash
#SBATCH --job-name=vl1_10_run
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL1-10 / VL1-11 (FROZEN protocol reports/VL1/VL1_FROZEN_PROTOCOL.md): one Table-A job; arguments passed to scripts/vl1_10_fit.py.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16; rc=0
while [ $# -gt 0 ]; do args=(); while [ $# -gt 0 ] && [ "$1" != "--then" ]; do args+=("$1"); shift; done; [ "$1" = "--then" ] && shift
  "$PYTHON" scripts/vl1_10_fit.py "${args[@]}" --out $OUTPUT_ROOT/VL1_10 || rc=1; done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
