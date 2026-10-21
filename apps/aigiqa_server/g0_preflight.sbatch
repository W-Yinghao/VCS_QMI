#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --partition=L40S
#SBATCH --exclude=node51,node53,node60
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# G0 CUDA preflight on FIT (package INSTALL section 4 / HANDOFF section 7): one job per GPU type (cross-hardware gate).
# Writes only $AIG_OUT/g0/<partition>_<jobid>/ on home (owner 2026-10-20), incl. the transient ~1.8 GB best.pt-shaped probe,
# plus a copy of G0_RESULT.json in the worktree evidence dir. Reads data/weights from /projects.
set -uo pipefail
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_aigiqa/apps/aigiqa_server/common.sh
OUT=$AIG_OUT/g0/${SLURM_JOB_PARTITION}_${SLURM_JOB_ID}
mkdir -p "$AIG_OUT/g0" "$AIG_SERVER/evidence/g0"
aig_slurm_record "$AIG_OUT/g0/${SLURM_JOB_PARTITION}_${SLURM_JOB_ID}.slurm.json" start
"$AIG_PY" "$AIG_SERVER/freeze_env.py" check --lock "$AIG_SERVER/ENV_LOCK.json" || exit $?
"$AIG_PY" "$AIG_SERVER/gpu_health.py" "$AIG_OUT/g0/${SLURM_JOB_PARTITION}_${SLURM_JOB_ID}.gpu_health.json" || exit $?
cd "$AIG_PKG"
"$AIG_PY" "$AIG_SERVER/g0_preflight.py" --base-config "$AIG_PROTOCOL/base_alignment.server.json" --out "$OUT"
rc=$?
aig_slurm_record "$AIG_OUT/g0/${SLURM_JOB_PARTITION}_${SLURM_JOB_ID}.slurm.json" end "$rc"
[ -f "$OUT/G0_RESULT.json" ] && cp "$OUT/G0_RESULT.json" "$AIG_SERVER/evidence/g0/G0_RESULT_${SLURM_JOB_PARTITION}_${SLURM_JOB_ID}.json"
exit "$rc"
