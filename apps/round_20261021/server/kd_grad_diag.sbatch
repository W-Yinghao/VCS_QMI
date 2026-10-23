#!/bin/bash
#SBATCH --job-name=r21_kd_chk_graddiag
#SBATCH --qos=normal
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --nice=0
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# KD checkpoint gradient diagnostic (kd_grad_diag.py, server-side addition): previous round's selected dkd_vcs_exact
# and dkd_bce checkpoints, 32 fixed FIT batches, CPU. FIT only; no TEST.
set -uo pipefail
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_round21/apps/round_20261021/server/common.sh
PY=$KD_PY
REC=$R21_REC/kd/grad_diag/slurm_${SLURM_JOB_ID:-manual}; mkdir -p "$REC"; cd "$REC" || exit 2
r21_record "$REC/slurm.json" start
"$PY" "$KD_ENV_CHECK" --out env_check.json > /dev/null || { r21_record "$REC/slurm.json" end 3; exit 3; }
"$PY" "$R21_SERVER/kd_grad_diag.py" --stage "$R21_STAGE" \
  --freeze /home/infres/yinwang/CS_QMI/outputs_apps/kd_20261020/protocol/kd_final_release.native_freeze.json \
  --threads "${SLURM_CPUS_PER_TASK:-8}" \
  --out "$R21_WT/apps/round_20261021/records/kd_grad_diag/KD_CHECKPOINT_GRADIENT_DIAGNOSTIC.json"
rc=$?
r21_record "$REC/slurm.json" end "$rc"
exit "$rc"
