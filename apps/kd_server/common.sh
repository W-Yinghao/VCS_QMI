#!/bin/bash
# Shared settings for the KD server wrappers (sourced once at job start). Absolute paths only.
# Package commands always run with cwd = KD_ROOT, so the package configs' relative paths
# (data/cifar100, protocol/..., external/..., runs/teacher_*) resolve inside the outputs tree.
export KD_WORKTREE=/home/infres/yinwang/CS_QMI/ssl_pilot_app_kd
export KD_PKG=$KD_WORKTREE/apps/VCS_QMI_KD_AIGIQA_Server_Package_v1_20261019
export KD_SERVER=$KD_WORKTREE/apps/kd_server
export KD_ROOT=/home/infres/yinwang/CS_QMI/outputs_apps/kd_20261020
export KD_PY=/home/infres/yinwang/CS_QMI/env_kd/bin/python
export PYTHONPATH="$KD_PKG:$KD_PKG/shared"   # as in README_CN / EXPERIMENT_LOCK_CN
export PYTHONDONTWRITEBYTECODE=1              # keep the package tree byte-identical (no __pycache__)
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
kd_ledger() { "$KD_PY" "$KD_SERVER/tools/ledger.py" "$@"; }
