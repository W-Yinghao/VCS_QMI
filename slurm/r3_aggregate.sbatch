#!/usr/bin/env bash
#SBATCH --job-name=r3_aggregate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r3 aggregation jobs (analysis only):  sbatch slurm/r3_aggregate.sbatch <p149|p150|p152>
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
case $1 in
  p149) "$PYTHON" scripts/p149_aggregate.py --out reports/P149_results.json; rc=$? ;;
  p150) "$PYTHON" scripts/p150_aggregate.py --out reports/P150_results.json; rc=$? ;;
  p152) "$PYTHON" scripts/p152_aggregate.py --out reports/P152_results.json; rc=$? ;;
  *) echo "unknown $1"; rc=2 ;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
