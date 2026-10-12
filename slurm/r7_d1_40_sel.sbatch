#!/usr/bin/env bash
#SBATCH --job-name=r7_d1_40_sel
#SBATCH --partition=RTX6000PRO,H100
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-40b selector units: torch only, main env (torch 2.9 cu128).  RTX6000PRO / H100 only (D1-20 execution note E5: one hardware class; real-data
# selector batches exceeded an L40S).  Resumable per unit (state saved at every CAL evaluation); the unit's marker <tag>.done.json is written
# last.  First arg: smoke | train <objective> <lr> <seed>.  Positional arguments only (no --export).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
R40=/home/infres/yinwang/CS_QMI/outputs/R7_D1/r7_40; D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; export PYTHONNOUSERSITE=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts slurm | wc -l)"
echo "code: $(cat scripts/r7_d1_40_*.py tests/test_r7_d1_40.py slurm/r7_d1_40_* | sha256sum | cut -c1-16) (sha256 of the R7-40 scripts, tests and wrappers)"
what="$1"; shift
case "$what" in
  smoke) "$PYTHON" -I scripts/r7_d1_40_train.py smoke --probe-records "$R40/smoke/probe/probe_records.jsonl" --cache "$R40/cache_smoke" \
           --out "$R40/train_smoke" "$@"; rc=$?;;
  train) obj="$1"; lr="$2"; seed="$3"; shift 3
         "$PYTHON" -I scripts/r7_d1_40_train.py train --objective "$obj" --lr "$lr" --seed "$seed" --teacher-dir "$R40/teacher" --manifest-dir "$D1/data" \
           --cache "$R40/cache" --out "$R40/selectors" "$@"; rc=$?;;
  *) echo "usage: smoke | train <objective> <lr> <seed>"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
