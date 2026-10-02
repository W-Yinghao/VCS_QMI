#!/usr/bin/env bash
#SBATCH --job-name=p113_diag
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P113 mechanism diagnostic (P111 hypothesis): scripts/p104_diagnostics.py on epoch_800.pt of each run in RUNS (space-separated) -> reports/P113_diag/<run>.json.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${RUNS:?}"; mkdir -p reports/P113_diag; rc=0
banner "P113 diag: $RUNS"
for r in $RUNS; do [ -f reports/P113_diag/$r.json ] && { echo "[skip] $r"; continue; }
  "$PYTHON" scripts/p104_diagnostics.py --run-dir /home/infres/yinwang/CS_QMI/outputs/$r --checkpoints epoch_800.pt --out reports/P113_diag/$r.json || rc=$?; done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
