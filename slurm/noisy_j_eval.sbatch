#!/usr/bin/env bash
#SBATCH --job-name=noisy_j_eval
#SBATCH --partition=RTX6000PRO,H100,A100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P104 addendum 2: evaluation-only clean / noisy held-out J.  Env: RUNS (space-separated run ids under outputs/), R_EVAL (default 16).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${RUNS:?}"
banner "noisy-J eval: $RUNS R_eval=${R_EVAL:-16}"
rc=0
for r in $RUNS; do
  "$PYTHON" scripts/noisy_j_eval.py --run-dir /home/infres/yinwang/CS_QMI/outputs/$r --r-eval "${R_EVAL:-16}" --out reports/P104_noisyJ_${r}.json || rc=$?
done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
