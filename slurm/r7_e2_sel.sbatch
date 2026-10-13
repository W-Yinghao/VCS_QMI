#!/usr/bin/env bash
#SBATCH --job-name=r7_e2_sel
#SBATCH --partition=RTX6000PRO,H100
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r7 E2 selector units: the frozen R7-40b trainer (scripts/r7_d1_40_train.py train; main env, torch 2.9 cu128) on E2's teacher records, manifest and
# cache.  RTX6000PRO / H100 only (40b_2 / 40b_3: the direct objective overflows an L40S already at 10k; at FIT_FULL the static data alone is ~25 GiB).
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (allocator layout only, no numerical effect; E2-11 = D1-FULL's DF-11).  Resumable per unit (state at
# every CAL evaluation; the marker <tag>.done.json is written last).  First arg: train <objective> <lr> <seed> | memsmoke [noexp] [...].
# Positional arguments only (no --export).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
E2=/home/infres/yinwang/CS_QMI/outputs/R7_E2; export PYTHONNOUSERSITE=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts slurm tests | wc -l)"
echo "code: E2 $(cat scripts/r7_e2_*.py tests/test_r7_e2.py slurm/r7_e2_* 2>/dev/null | sha256sum | cut -c1-16), frozen R7-40 $(cat scripts/r7_d1_40_*.py | sha256sum | cut -c1-16)"
what="$1"; shift
case "$what" in
  train) [[ $# -eq 3 && "$1" =~ ^(vcs|logistic|direct)$ && "$2" =~ ^[0-9.e-]+$ && "$3" =~ ^[0-9]+$ ]] || { echo "train <vcs|logistic|direct> <lr> <seed>"; exit 2; }
         export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
         "$PYTHON" -I scripts/r7_d1_40_train.py train --objective "$1" --lr "$2" --seed "$3" --teacher-dir "$E2/teacher" --manifest-dir "$E2/data" \
           --cache "$E2/cache" --out "$E2/selectors"; rc=$?;;
  memsmoke) if [ "${1:-}" = "noexp" ]; then shift; else export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; fi
            "$PYTHON" -I scripts/r7_e2_smoke.py memsmoke --out "$E2/smoke/memsmoke" "$@"; rc=$?;;
  *) echo "usage: train <vcs|logistic|direct> <lr> <seed> | memsmoke [noexp] [...]"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
