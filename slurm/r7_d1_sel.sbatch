#!/usr/bin/env bash
#SBATCH --job-name=r7_d1_sel
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-30 D1-20 selector units: torch only, main env (torch 2.9 cu128) -> any of the three partitions incl. Blackwell.  Resumable per unit
# (state saved at every CAL evaluation; a rerun continues from the last save).  First arg: smoke | train <objective> <lr> <seed> [--no-question]
# [--target d].  Positional arguments only (no --export).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; export PYTHONNOUSERSITE=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ)"
what="$1"; shift
case "$what" in
  smoke) "$PYTHON" scripts/r7_d1_train.py smoke --probe-records "$D1/smoke/probe/probe_records.jsonl" --cache "$D1/cache_smoke" --out "$D1/train_smoke" "$@"; rc=$?;;
  train) obj="$1"; lr="$2"; seed="$3"; shift 3
         "$PYTHON" scripts/r7_d1_train.py train --objective "$obj" --lr "$lr" --seed "$seed" --teacher-dir "$D1/teacher" --manifest-dir "$D1/data" \
           --cache "$D1/cache" --out "$D1/selectors" "$@"; rc=$?;;
  *) echo "usage: smoke | train <objective> <lr> <seed> [--no-question] [--target d]"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
