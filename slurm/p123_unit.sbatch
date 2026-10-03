#!/usr/bin/env bash
#SBATCH --job-name=p123_grad
#SBATCH --partition=A100,L40S,RTX6000PRO,H100
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P123 (v6 V6-GRAD) full run, one bundle job: RUNS (comma list, via the environment), EXTRA (e.g. --grad-batches 8).  Resumable (finished run
# JSONs are skipped).  Read-only on checkpoints; normal QOS only (never runfill); node60 allowed.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${RUNS:?}"
banner "P123 V6-GRAD: $RUNS ${EXTRA:-}"
"$PYTHON" scripts/p123_grad.py --runs "$RUNS" ${EXTRA:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
