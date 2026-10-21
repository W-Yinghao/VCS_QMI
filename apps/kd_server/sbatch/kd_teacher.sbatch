#!/bin/bash
# S2: strict FIT-only teacher with the package's native `kd train-teacher` (fixed final epoch, CAL only logged).
# Args: <package config json> <run dir relative to KD_ROOT, e.g. runs/teacher_wrn40_2>
# Gate first: env lock + functional CUDA check + G0 gpu_smoke/cuda_math on THIS GPU type; no training if it fails.
# Preemption-safe: resubmitting the same line resumes from the epoch-boundary last.pt (teacher_tool.py mode);
# an unusable partial run is renamed aside, never deleted. After success, teacher.pt/provenance are made
# read-only and last.pt is deleted (storage rule; teacher.pt holds the final-epoch weights).
set -uo pipefail
CFG="$1"; OUT="$2"
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_kd/apps/kd_server/common.sh
export KD_BATCH_SCRIPT="$0"
cd "$KD_ROOT" || exit 2
TAG=$(basename "$OUT"); JOB=${SLURM_JOB_ID:-manual}
REC="$KD_ROOT/ledger/teacher/${TAG}__slurm${JOB}"
kd_ledger start --kind teacher --tag "$TAG"
trap 'echo "SIGTERM at $(date -u +%FT%TZ): resubmit the same line to resume from last.pt"; kd_ledger end --kind teacher --tag "$TAG" --rc 143 --stage sigterm; exit 143' TERM
if ! "$KD_PY" "$KD_SERVER/tools/env_check.py" --cuda --out "$REC.env.json" > /dev/null; then
  kd_ledger end --kind teacher --tag "$TAG" --rc 3 --stage env_check; exit 3
fi
if ! "$KD_PY" "$KD_SERVER/g0/g0_driver.py" --out "$KD_ROOT/g0/teacher_prelude_${TAG}_slurm${JOB}" --device cuda --stages env,gpu_smoke,cuda_math; then
  kd_ledger end --kind teacher --tag "$TAG" --rc 5 --stage gpu_smoke_gate; exit 5
fi
MODE=$("$KD_PY" "$KD_SERVER/tools/teacher_tool.py" mode "$OUT") || { kd_ledger end --kind teacher --tag "$TAG" --rc 6 --stage mode; exit 6; }
case "$MODE" in
  complete) echo "teacher already complete: $OUT"; kd_ledger end --kind teacher --tag "$TAG" --rc 0 --stage already_complete; exit 0;;
  resume) EXTRA=(--resume);;
  fresh) EXTRA=();;
  *) echo "unexpected mode: $MODE"; kd_ledger end --kind teacher --tag "$TAG" --rc 6 --stage mode; exit 6;;
esac
echo "teacher mode: $MODE"
"$KD_PY" -m kd train-teacher --config "$CFG" --out "$OUT" "${EXTRA[@]}"; rc=$?
"$KD_PY" "$KD_SERVER/tools/teacher_tool.py" post "$OUT" "$rc" "$REC.post.json"; prc=$?
[ "$rc" -eq 0 ] && rc=$prc
kd_ledger end --kind teacher --tag "$TAG" --rc "$rc" --stage train_teacher --extra "{\"mode\": \"$MODE\"}"
exit "$rc"
