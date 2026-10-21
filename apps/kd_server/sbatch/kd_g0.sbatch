#!/bin/bash
# S3: G0 CUDA preflight for ONE GPU type (submit once per partition; see apps/kd_server/g0/G0_PLAN.json).
# Arg: <type tag, e.g. rtx6000pro | l40s | h100>.  Needs the strict FIT teacher runs/teacher_wrn40_2 (S2).
set -uo pipefail
TAG="$1"
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_kd/apps/kd_server/common.sh
export KD_BATCH_SCRIPT="$0"
cd "$KD_ROOT" || exit 2
JOB=${SLURM_JOB_ID:-manual}
kd_ledger start --kind g0 --tag "$TAG"
trap 'echo "SIGTERM at $(date -u +%FT%TZ)"; kd_ledger end --kind g0 --tag "$TAG" --rc 143 --stage sigterm; exit 143' TERM
if ! "$KD_PY" "$KD_SERVER/tools/env_check.py" --cuda --out "$KD_ROOT/ledger/g0/${TAG}__slurm${JOB}.env.json" > /dev/null; then
  kd_ledger end --kind g0 --tag "$TAG" --rc 3 --stage env_check; exit 3
fi
"$KD_PY" "$KD_SERVER/g0/g0_driver.py" --out "$KD_ROOT/g0/${TAG}_slurm${JOB}" --device cuda; rc=$?
kd_ledger end --kind g0 --tag "$TAG" --rc "$rc" --stage g0
exit "$rc"
