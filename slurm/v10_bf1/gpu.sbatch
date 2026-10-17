#!/usr/bin/env bash
#SBATCH --job-name=v10bf_gpu
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --nice=0
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC
job_selector=${SLURM_JOB_ID:?missing Slurm job}
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then job_selector="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; fi
export V10_BF1_SCHED_START=$(squeue -j "$job_selector" -h -o '%S')
export V10_BF1_JOB_START=$(/home/infres/yinwang/CS_QMI/env/bin/python -c 'import os,datetime; s=os.environ["V10_BF1_SCHED_START"].strip(); print(datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc).timestamp())')
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1 TZ=UTC
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
R=/home/infres/yinwang/CS_QMI/handoff/v10_bf1_20261017/source
export PYTHONPATH="$R/src:$R"
export V10_SUBMIT_COMMIT="${V10_BF1_SOURCE_COMMIT:?missing source commit}"
export V10_EXPECT_CODE=e08ee2c176475e2b
cd "$R"
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
child=''
forward_signal() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap forward_signal USR1 TERM
if [ "${1:?gate|cal|final|array}" = gate ]; then
  "$PYTHON" scripts/v10_bf1/check_control.py
  "$PYTHON" scripts/v10_bf1/gpu_gate.py &
elif [ "$1" = array ]; then
  case "${SLURM_ARRAY_TASK_ID:?missing array index}" in
    0) family=MF_VCS_perm1 ;;
    1) family=MF_VCS_M0 ;;
    2) family=MF_VCS_M1 ;;
    3) family=MLP_VCS_perm1 ;;
    *) exit 2 ;;
  esac
  "$PYTHON" scripts/v10_bf1/run.py all "$family" &
else
  "$PYTHON" scripts/v10_bf1/run.py "$@" &
fi
child=$!
set +e
wait "$child"
rc=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; rc=$?; fi
export V10_BF1_EXIT_CODE=$rc
"$PYTHON" - <<'PY'
import json,os,time
from pathlib import Path
p=Path('/home/infres/yinwang/CS_QMI/handoff/v10_bf1_20261017/exits')
p.mkdir(parents=True,exist_ok=True)
r={'job_id':os.environ['SLURM_JOB_ID'],'exit_code':int(os.environ['V10_BF1_EXIT_CODE']),
   'start_unix':float(os.environ['V10_BF1_JOB_START']),'end_unix':time.time()}
r['allocated_seconds']=r['end_unix']-r['start_unix']
(p/(r['job_id']+'.json')).write_text(json.dumps(r,indent=2)+'\n')
PY
exit "$rc"
