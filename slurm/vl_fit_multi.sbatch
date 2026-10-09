#!/usr/bin/env bash
#SBATCH --job-name=vl_fit_multi
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Several Table A feature sets in one CPU job (cap relief).  Groups separated by --set "<VL_DATASET> <VL1_OUT>"; within a group, fitter
# argument lists separated by --then.  Each group runs the frozen fitter exactly as slurm/vl1_10_run.sbatch would.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16; rc=0
while [ $# -gt 0 ]; do
  [ "$1" = "--set" ] || { echo "expected --set"; exit 2; }; export VL_DATASET=$2 VL1_OUT=$3; shift 3; echo "== $VL_DATASET -> $VL1_OUT"
  while [ $# -gt 0 ] && [ "$1" != "--set" ]; do args=(); while [ $# -gt 0 ] && [ "$1" != "--then" ] && [ "$1" != "--set" ]; do args+=("$1"); shift; done; [ "$1" = "--then" ] && shift
    "$PYTHON" scripts/vl1_10_fit.py "${args[@]}" --out $OUTPUT_ROOT/$VL1_OUT || rc=1; done
done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
