#!/usr/bin/env bash
#SBATCH --job-name=r7_d1full_sel
#SBATCH --partition=RTX6000PRO,H100
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-30 D1-FULL selector units: the frozen D1-20 trainer (main env, torch 2.9 cu128) on the D1-FULL inputs.  RTX6000PRO / H100 only (D1-20
# execution note E5; the frozen trainer keeps all FIT_FULL features on the GPU, ~38.6 GiB, so an L40S cannot hold a unit).  Resumable per unit.
# First arg: train <objective> <lr> <seed> [--no-question] [--target d] | memsmoke [...].  Positional arguments only (no --export).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
DF=/home/infres/yinwang/CS_QMI/outputs/R7_D1_FULL; export PYTHONNOUSERSITE=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ)"
what="$1"; shift
case "$what" in
  train) export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True       # allocator layout only, no numerical effect: peak reserved 89.6 -> 77.5 GiB
         "$PYTHON" scripts/r7_d1full_train.py train "$@" --root "$DF"; rc=$?;;   # at FIT_FULL scale (memory smokes 1035986 / 1036018; OPEN DF-11)
  memsmoke) "$PYTHON" scripts/r7_d1full_train.py memsmoke "$@"; rc=$?;;
  *) echo "usage: train <objective> <lr> <seed> [--no-question] [--target d] | memsmoke"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
