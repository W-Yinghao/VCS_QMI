#!/usr/bin/env bash
#SBATCH --job-name=p152_run
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
#SBATCH --time=06:00:00
# P152 (FROZEN prereg): one dependence level, seeds 0-2, all six in-batch methods.  Units whose JSON exists are skipped.
#   sbatch slurm/p152_run.sbatch <I>
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
I=$1; O=$OUTPUT_ROOT/P152_e1; rc=0
for s in 0 1 2; do
  [ -f $O/I${I}_s${s}.json ] && [ "$("$PYTHON" -c "import json;print(len(json.load(open('$O/I${I}_s${s}.json'))['kinds']))")" = 6 ] && { echo "skip I$I s$s"; continue; }
  "$PYTHON" scripts/p152_e1_stability.py --I $I --seed $s --out $O || rc=1
done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
