#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# One fixed matrix row through the package entry (no edits):
#   python <package>/apps/run_one.py --package-root <stage> --matrix <JOBS.csv> --index <i>
# Usage: sbatch --job-name=... --cpus-per-task=.. --mem=.. --time=.. run_one.sbatch <aigiqa|kd> <index>
# Env check and GPU check run BEFORE run_one.py creates the run dir, so an infrastructure fault (exit 3/4) leaves the
# fixed run dir unused and the same line can be resubmitted. The trainers have no resume: a failure after the run dir
# exists needs a new run root (make_configs.py), recipe and seed unchanged (APPS_SERVER_CN §7).
set -uo pipefail
APP=${1:?aigiqa|kd}; IDX=${2:?index}
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_round21/apps/round_20261021/server/common.sh
case "$APP" in
  aigiqa) PY=$AIG_PY; MATRIX=$R21_APPS/configs/aigiqa/JOBS.csv
          export HF_HOME=$AIG_HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 ;;
  kd)     PY=$KD_PY;  MATRIX=$R21_APPS/configs/kd_probe/JOBS.csv ;;
  *) echo "unknown app $APP"; exit 2 ;;
esac
ROW=$(awk -F, -v i=$((IDX + 2)) 'NR==i {print $2"__seed"$3}' "$MATRIX")
[ -n "$ROW" ] || { echo "no row $IDX in $MATRIX"; exit 2; }
REC=$R21_REC/$APP/$ROW/slurm_${SLURM_JOB_ID:-manual}
mkdir -p "$REC"
r21_record "$REC/slurm.json" start
stop() { r21_record "$REC/slurm.json" end "$1"; echo "pre-run stop rc=$1 ($2); run dir untouched" >&2; exit "$1"; }
cd "$REC" || exit 2
if [ "$APP" = aigiqa ]; then
  "$PY" "$AIG_ENV_TOOL" check --lock "$AIG_ENV_LOCK" > "$REC/env_check.json" 2> "$REC/env_check.err" || stop 3 env_drift
  "$PY" "$AIG_GPU_HEALTH" "$REC/gpu_health.json" > /dev/null || stop 4 gpu_health
else
  "$PY" "$KD_ENV_CHECK" --cuda --out "$REC/env_check.json" > /dev/null || stop 3 env_or_cuda_check
fi
git -C "$R21_WT" rev-parse HEAD > server_git_head
sha256sum "$R21_STAGE/APPS_OVERLAY_INSTALL.json" > stage_install.sha256
echo "app=$APP index=$IDX row=$ROW node=${SLURMD_NODENAME:-?} partition=${SLURM_JOB_PARTITION:-?} start=$(date -u +%FT%TZ)"
"$PY" "$R21_APPS/run_one.py" --package-root "$R21_STAGE" --matrix "$MATRIX" --index "$IDX"
rc=$?
r21_record "$REC/slurm.json" end "$rc"
echo "end=$(date -u +%FT%TZ) rc=$rc"
exit "$rc"
