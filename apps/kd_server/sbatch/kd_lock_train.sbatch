#!/bin/bash
# S5/S7: one globally locked KD candidate through the package entry point
#   python -m vcs_common.experiment train --lock <lock> --job-id <job_id>
# Args: <lock json> <job_id from `list-jobs`>
# 1. env lock + functional CUDA check, 2. preflight of the exact native path on this node/GPU (lock check,
#    receipt absent, fresh run dir, data, teacher, model, loss, one real FIT batch) - both BEFORE the receipt,
#    so infrastructure faults leave the job ID unused (exit 3/4: resubmit the same line),
# 3. global train (no resume exists here: an interrupted/failed candidate needs an amendment, EXPERIMENT_LOCK §6),
# 4. post: receipt classification; COMPLETED -> record sizes/hashes, delete last.pt (storage rule).
set -uo pipefail
LOCK="$1"; JOB_ID="$2"
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_kd/apps/kd_server/common.sh
export KD_BATCH_SCRIPT="$0"
cd "$KD_ROOT" || exit 2
JOB=${SLURM_JOB_ID:-manual}
KIND=$(basename "$LOCK" .json)
REC="$KD_ROOT/ledger/$KIND/${JOB_ID}__slurm${JOB}"
kd_ledger start --kind "$KIND" --tag "$JOB_ID" --lock "$LOCK"
on_term() {  # time limit / scancel: the python child dies without updating its receipt; classify and record
  echo "SIGTERM at $(date -u +%FT%TZ): receipt left as is; see $REC.post.json"
  "$KD_PY" "$KD_SERVER/tools/post_lock_job.py" --lock "$LOCK" --job-id "$JOB_ID" --train-rc 143 --out "$REC.post.json"
  kd_ledger end --kind "$KIND" --tag "$JOB_ID" --rc 143 --stage sigterm
  exit 143
}
trap on_term TERM
if ! "$KD_PY" "$KD_SERVER/tools/env_check.py" --cuda --out "$REC.env.json" > /dev/null; then
  kd_ledger end --kind "$KIND" --tag "$JOB_ID" --rc 3 --stage env_check; exit 3
fi
if ! "$KD_PY" "$KD_SERVER/tools/preflight_lock_job.py" --lock "$LOCK" --job-id "$JOB_ID" --out "$REC.preflight.json"; then
  kd_ledger end --kind "$KIND" --tag "$JOB_ID" --rc 4 --stage preflight; exit 4
fi
"$KD_PY" -m vcs_common.experiment train --lock "$LOCK" --job-id "$JOB_ID"; rc=$?
"$KD_PY" "$KD_SERVER/tools/post_lock_job.py" --lock "$LOCK" --job-id "$JOB_ID" --train-rc "$rc" --out "$REC.post.json"
kd_ledger end --kind "$KIND" --tag "$JOB_ID" --rc "$rc" --stage train
exit "$rc"
