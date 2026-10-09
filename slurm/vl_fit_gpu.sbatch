#!/usr/bin/env bash
#SBATCH --job-name=vl_fit_gpu
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Several Table A feature sets in one CPU job (cap relief).  Groups separated by --set "<VL_DATASET> <VL1_OUT>"; within a group, fitter
# argument lists separated by --then.  Each group runs the frozen fitter exactly as slurm/vl1_10_run.sbatch would.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 VL_DEVICE=cuda; rc=0   # neural routes on the GPU (equivalence gate: reports/VL1/vl_gpu_fit_check.json)
while [ $# -gt 0 ]; do
  [ "$1" = "--set" ] || { echo "expected --set"; exit 2; }; export VL_DATASET=$2 VL1_OUT=$3; shift 3; echo "== $VL_DATASET -> $VL1_OUT"
  while [ $# -gt 0 ] && [ "$1" != "--set" ]; do args=(); while [ $# -gt 0 ] && [ "$1" != "--then" ] && [ "$1" != "--set" ]; do args+=("$1"); shift; done; [ "$1" = "--then" ] && shift
    "$PYTHON" scripts/vl1_10_fit.py "${args[@]}" --out $OUTPUT_ROOT/$VL1_OUT || rc=1; done
done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
