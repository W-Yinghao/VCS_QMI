#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --nice=50
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# One registered AGIQA-3K alignment pilot candidate via the package's global lock:
#   python -m vcs_common.experiment train --lock LOCK --job-id <registered id>
# Usage: sbatch [options as in SUBMIT_LINES.sh] train_job.sbatch <config stem, e.g. vcs_exact_text_w0.1_s3101>
# No resume exists (native trainer + lock refuse a second attempt); hence --no-requeue and the pre-lock env/GPU checks:
# a failure before `experiment train` leaves no receipt and the same line can simply be resubmitted.
set -uo pipefail
NAME=${1:?config stem}
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_aigiqa/apps/aigiqa_server/common.sh
REC=$AIG_RECORDS/runs/${NAME}/slurm_${SLURM_JOB_ID}
mkdir -p "$REC"
aig_slurm_record "$REC/slurm.json" start
fail() { aig_slurm_record "$REC/slurm.json" end "$1"; echo "pre-lock stop rc=$1 ($2); no receipt written" >&2; exit "$1"; }
"$AIG_PY" "$AIG_SERVER/freeze_env.py" check --lock "$AIG_SERVER/ENV_LOCK.json" > "$REC/env_check.json" 2>/dev/null || fail $? env_drift
"$AIG_PY" "$AIG_SERVER/gpu_health.py" "$REC/gpu_health.json" > /dev/null || fail $? gpu_health
JOB_ID=$("$AIG_PY" "$AIG_SERVER/resolve_job.py" --lock "$AIG_LOCK" --name "$NAME") || fail $? resolve
echo "$JOB_ID" > "$REC/job_id"
git -C "$AIG_REPO" rev-parse HEAD > "$REC/server_git_head"
echo "name=$NAME job_id=$JOB_ID node=${SLURMD_NODENAME:-?} partition=${SLURM_JOB_PARTITION:-?} start=$(date -u +%FT%TZ)"
cd "$AIG_PKG"
"$AIG_PY" -m vcs_common.experiment train --lock "$AIG_LOCK" --job-id "$JOB_ID"
rc=$?
aig_slurm_record "$REC/slurm.json" end "$rc"
echo "end=$(date -u +%FT%TZ) rc=$rc"
exit "$rc"
