#!/usr/bin/env bash
#SBATCH --job-name=v10val
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --nice=500
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
: "${V10_VALIDATION_SOURCE:?immutable source required}"
: "${V10_VALIDATION_MANIFEST:?validation manifest required}"
: "${V10_VALIDATION_SHA256:?submission pin required}"
export PYTHONPATH="$V10_VALIDATION_SOURCE/src:$V10_VALIDATION_SOURCE"
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
export V10_SUPPLEMENT_JOB_START
V10_SUPPLEMENT_JOB_START=$("$PYTHON" -c 'import time; print(time.time())')
cd "$V10_VALIDATION_SOURCE"
child=''
forward_signal() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap forward_signal USR1 TERM
"$PYTHON" -u scripts/v10_supplement/validate.py "$@" &
child=$!
set +e
wait "$child"
rc=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; rc=$?; fi
export V10_VALIDATION_EXIT=$rc
"$PYTHON" - <<'PY'
import json,os,time
from pathlib import Path
b=Path(os.environ['V10_VALIDATION_MANIFEST']).parent/'exits'; b.mkdir(exist_ok=True)
r={'job':os.environ['SLURM_JOB_ID'],'exit_code':int(os.environ['V10_VALIDATION_EXIT']),
   'validation_manifest_sha256':os.environ['V10_VALIDATION_SHA256'],
   'start_unix':float(os.environ['V10_SUPPLEMENT_JOB_START']),'end_unix':time.time()}
r['batch_seconds']=r['end_unix']-r['start_unix']
p=b/(r['job']+'.json'); t=p.with_suffix('.tmp'); t.write_text(json.dumps(r,indent=2)+'\n'); t.replace(p)
PY
receipt_rc=$?
if [ "$rc" -eq 0 ] && [ "$receipt_rc" -ne 0 ]; then rc=$receipt_rc; fi
exit "$rc"
