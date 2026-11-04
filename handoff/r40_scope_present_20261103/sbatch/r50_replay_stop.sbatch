#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103/logs/%x_%j.out
# R50 JOINT replay with the full 8000-update schedule stopped at 250 (tools/r50_replay_stop.py <tree> <units> <out>); node pinned to the reference unit's host
set -uo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103; PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"   # R50 amendment A2 tie policy
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name,driver_version --format=csv; echo "start $(date -u +%FT%TZ) $(hostname) $1"
$PY -u -B $W/tools/r50_replay_stop.py "$1" "$2" /home/infres/yinwang/CS_QMI/outputs_v9_r50/r50_gate/${SLURM_JOB_ID}_stop; echo "R50_SMOKE_END rc=$? $(date -u +%FT%TZ)"
