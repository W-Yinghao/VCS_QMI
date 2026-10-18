#!/usr/bin/env bash
#SBATCH --job-name=v10supp
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
: "${SLURM_JOB_ID:?Slurm required}"
: "${V10_SUPPLEMENT_SOURCE:?explicit committed source clone required}"
: "${V10_SUPPLEMENT_EXPECT_RELEASE:?validated execution release hash required}"
: "${V10_SUPPLEMENT_RELEASE:?validated execution release path required}"
export PYTHONPATH="$V10_SUPPLEMENT_SOURCE/src:$V10_SUPPLEMENT_SOURCE"
export V10_SUPPLEMENT_SCHED_START
V10_SUPPLEMENT_SCHED_START=$(squeue -j "$SLURM_JOB_ID" -h -o '%S')
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
export V10_SUPPLEMENT_JOB_START
V10_SUPPLEMENT_JOB_START=$("$PYTHON" -c 'import os,datetime; s=os.environ["V10_SUPPLEMENT_SCHED_START"].strip(); print(datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc).timestamp())')
cd "$V10_SUPPLEMENT_SOURCE"
child=''
forward_signal() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap forward_signal USR1 TERM
"$PYTHON" scripts/v10_supplement/run.py "$@" --release "$V10_SUPPLEMENT_RELEASE" &
child=$!
set +e
wait "$child"
rc=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; rc=$?; fi
export V10_SUPPLEMENT_EXIT_CODE=$rc
"$PYTHON" - <<'PY'
import json,os,time
from pathlib import Path
release=Path(os.environ['V10_SUPPLEMENT_RELEASE']).resolve()
p=release.parent/'exits';p.mkdir(parents=True,exist_ok=True)
r={'job':os.environ['SLURM_JOB_ID'],'exit_code':int(os.environ['V10_SUPPLEMENT_EXIT_CODE']),
   'start_unix':float(os.environ['V10_SUPPLEMENT_JOB_START']),'end_unix':time.time(),
   'release_sha256':os.environ['V10_SUPPLEMENT_EXPECT_RELEASE']}
r['allocated_seconds']=r['end_unix']-r['start_unix']
tmp=p/('.'+r['job']+'.tmp');tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(p/(r['job']+'.json'))
PY
receipt_rc=$?
if [ "$rc" -eq 0 ] && [ "$receipt_rc" -ne 0 ]; then rc=$receipt_rc; fi
exit "$rc"
