#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# One r22 JOBS row:  sbatch --job-name=<row job_name> ... run_one.sbatch <code snapshot dir> <JOBS csv relative to it> [--grad-check]
# The snapshot dir is explicit in every line, so a queued job always runs the code it was submitted with.
# Env check and GPU check run BEFORE the run dir exists (exit 3/4: resubmit the same line). The trainers have no
# resume; a failure after the run dir exists needs a new run dir (recipe and seed unchanged).
set -uo pipefail
CODE=${1:?code snapshot dir}; MATRIX_REL=${2:?JOBS csv relative to the code snapshot}; EXTRA=${3:-}
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_r22/apps/r22/server/common.sh
MATRIX=$CODE/$MATRIX_REL
APP=$(awk -F, -v n="$SLURM_JOB_NAME" 'NR>1 && $6==n {print $1}' "$MATRIX")
case "$APP" in
  aigiqa) PY=$AIG_PY; export HF_HOME=$AIG_HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 ;;
  kd)     PY=$KD_PY ;;
  *) echo "job name $SLURM_JOB_NAME not in $MATRIX"; exit 2 ;;
esac
REC=$R22_REC/$SLURM_JOB_NAME/slurm_${SLURM_JOB_ID:-manual}; mkdir -p "$REC"
r22_record "$REC/slurm.json" start
stop() { r22_record "$REC/slurm.json" end "$1"; echo "pre-run stop rc=$1 ($2); run dir untouched" >&2; exit "$1"; }
trap 'r22_record "$REC/slurm.json" end 143; exit 143' TERM
cd "$REC" || exit 2
echo "$CODE" > code_dir; cat "$CODE/CODE_COMMIT" > code_commit 2>/dev/null
if [ "$APP" = aigiqa ]; then
  "$PY" "$AIG_ENV_TOOL" check --lock "$AIG_ENV_LOCK" > env_check.json 2> env_check.err || stop 3 env_drift
  "$PY" "$AIG_GPU_HEALTH" "$REC/gpu_health.json" > /dev/null || stop 4 gpu_health
else
  "$PY" "$KD_ENV_CHECK" --cuda --out "$REC/env_check.json" > /dev/null || stop 3 env_or_cuda_check
fi
echo "app=$APP job=$SLURM_JOB_NAME node=${SLURMD_NODENAME:-?} partition=${SLURM_JOB_PARTITION:-?} code=$CODE start=$(date -u +%FT%TZ)"
"$PY" -I "$CODE/run_one.py" --matrix "$MATRIX" --job-name "$SLURM_JOB_NAME" $EXTRA &
wait $!; rc=$?
r22_record "$REC/slurm.json" end "$rc"
echo "end=$(date -u +%FT%TZ) rc=$rc"
exit "$rc"
