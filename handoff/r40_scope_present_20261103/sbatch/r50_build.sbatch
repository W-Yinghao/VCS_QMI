#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103/logs/%x_%j.out
# R50 build (FIT_FIXED_CENTER_V1): tools/r50_build.py <tree> <base> [--h-only]. GPU type pinned to the R47 build GPU type of the base so the recomputed
# FIT encodings can be compared bitwise with the R47 G FIT cache (FIT identity evidence for the weight reuse).
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103; PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"   # R50 amendment A2 tie policy (R50_AMENDMENT_A2.json)
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version --format=csv; echo "start $(date -u +%FT%TZ) $(hostname) $*"
$PY -u -B $W/tools/r50_build.py "$@"
