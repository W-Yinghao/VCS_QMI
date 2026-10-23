#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R22-A / A2 GPU job.  Full unit:  sbatch --job-name=r22_vl[2]_<arm>_s<seed> [partition/time/nice] r22_vl.sbatch run SEED ARM
#                 Smoke:      sbatch --job-name=r22_vl_chk_<gpu>     [partition/time]      r22_vl.sbatch smoke SEED [ARMS] [STEPS]
# A full unit resumes from its last complete epoch when the identical line is resubmitted (time limit / node failure);
# a COMPLETE_VERIFIED unit is skipped.  Start/end/GPU are printed here and kept per attempt in runs/<unit>/attempts.jsonl.
set -u
cmd=${1:?usage: r22_vl.sbatch run|smoke SEED ...}; seed=${2:?seed}
D=${R22_DIR:-/home/infres/yinwang/CS_QMI/ssl_pilot_v9_r22a/v9_r22/vcsloss}   # R22_DIR: code directory override (A2 pre-merge checks)
ROOT=${R22_ROOT:-/home/infres/yinwang/CS_QMI/outputs_v9_r22/vcsloss}
PY=/home/infres/yinwang/CS_QMI/env/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONDONTWRITEBYTECODE=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) job=${SLURM_JOB_ID:-} partition=${SLURM_JOB_PARTITION:-} cmd=$* start=$(date -u +%FT%TZ) head=$(git -C "$D" rev-parse --short HEAD)"
if [ "$cmd" = run ]; then
  "$PY" -I -B "$D/r22_student.py" run --root "$ROOT" --seed "$seed" --arm "${3:?arm}"; rc=$?
else
  "$PY" -I -B "$D/r22_student.py" smoke --root "$ROOT" --seed "$seed" --arms "${3:-VCS,VCSW,BCE,NCE,HINGE}" --steps "${4:-200}"; rc=$?
fi
echo "end=$(date -u +%FT%TZ) rc=$rc"; exit $rc
