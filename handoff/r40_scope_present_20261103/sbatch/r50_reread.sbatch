#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103/logs/%x_%j.out
# R50 PRIMARY re-read (zero training) of R47 G / R48 N references: tools/r50_reread.py <tree> <units>
set -uo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103; PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"   # R50 amendment A2 tie policy (R50_AMENDMENT_A2.json)
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version --format=csv; echo "start $(date -u +%FT%TZ) $(hostname) $1"
$PY -u -B $W/tools/r50_reread.py "$1" "$2"; echo "R50_REREAD_JOB_END rc=$? $(date -u +%FT%TZ)"
