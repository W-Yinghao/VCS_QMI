#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103/logs/%x_%j.out
# R46 GPU smoke / gate (user: 「该提交的CPU，GPUgate smoke test现在就准备」): the production trainer in gate mode ($1 = tree, $2 = units), 20 updates per
# unit with readouts at 0 and 20, scratch dir r46_parents/gate/<job> (not formal, never reused): step-0 native reproduction (A) / round-1 step-8000
# DEV rank equality (B), R38 graph check, R39 / R40 one-step checks, frozen H / student tensors, pool sampler, round-2 tables. Then tools/r46_smoke_check.py.
set -uo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
G=/projects/EEG-foundation-model/yinghao/outputs_v9/r40_scope_present_20261103/r46_parents/gate/$SLURM_JOB_ID
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; echo "start $(date -u +%FT%TZ) $(hostname)"
$PY -u -B $W/$1/train_unit.py --units "$2" --gate-out $G --gate-steps 20; rc=$?
$PY -I -B $W/tools/r46_smoke_check.py $G "$2" $rc
