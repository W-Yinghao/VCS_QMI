#!/usr/bin/env bash
#SBATCH --job-name=v10d
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V10 post-DEV sensitivity diagnostics D1-D3 (pre-registered: v10/sensitivity/README_CN.md, PREREGISTRATION.json).
# usage: sbatch [--partition=...] --export=ALL,V10D_FREEZE_SHA256=<sha> gpu.sbatch d1 RUN_KEY | d2 | d3 FAMILY
# D1/D2 must run on L40S (as the frozen rows); D3 on RTX6000PRO or H100. The Python runners re-check all of it.
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
: "${SLURM_JOB_ID:?Slurm required}"
: "${V10D_FREEZE_SHA256:?submission pin of the bundle FREEZE.json required}"
DIAG_ROOT=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_diag
BUNDLE=/home/infres/yinwang/CS_QMI/handoff/v10_sensitivity_20261019
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
diag=${1:?diagnostic d1|d2|d3}; shift
case "$diag" in
  d1) script=d1_comir_checkpoints.py; allowed="L40S"
      source_root=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_native_recovery_20261019 ;;
  d2) script=d2_mind_wtd.py; allowed="L40S"
      source_root=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_pipeline_20261019 ;;
  d3) script=d3_gate.py; allowed="RTX6000PRO H100"
      source_root=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_pipeline_20261019
      export V10_VALIDATION_MANIFEST=$BUNDLE/D3/validation/MANIFEST.json
      export V10_VALIDATION_SHA256=8befdabb0c2b6823e8c04c86763090fd2b383b28b5c02250c34fcae15eb1292b ;;
  *) echo "[v10d] unknown diagnostic: $diag" >&2; exit 2 ;;
esac
export V10D_DIAGNOSTIC=$diag V10D_ARGS="$*"
export PYTHONPATH="$source_root/src:$source_root"
export V10D_BATCH_START V10D_SCHED_START V10_SUPPLEMENT_JOB_START
V10D_BATCH_START=$(date +%s.%N)
V10D_SCHED_START=$(squeue -j "$SLURM_JOB_ID" -h -o '%S' 2>/dev/null || true)
# Accepted-runner accounting: the Slurm StartTime (UTC under TZ=UTC); batch start only if squeue fails.
V10_SUPPLEMENT_JOB_START=$("$PYTHON" -c 'import datetime,os
s=os.environ.get("V10D_SCHED_START","").strip()
try: print(datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc).timestamp())
except ValueError: print(os.environ["V10D_BATCH_START"])')
echo "[v10d] diag=$diag args=$* job=$SLURM_JOB_ID partition=${SLURM_JOB_PARTITION:-?} host=$(hostname) utc=$(date -u +%FT%TZ) slurm_start=$V10D_SCHED_START"

receipt() {  # receipt STATUS EXIT_CODE
  V10D_STATUS=$1 V10D_EXIT=$2 "$PYTHON" - "$BUNDLE" <<'PY' || true
import json, os, subprocess, sys, time
from pathlib import Path
query = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True)
end = time.time(); start = float(os.environ["V10_SUPPLEMENT_JOB_START"])
r = {"job": os.environ["SLURM_JOB_ID"], "diagnostic": os.environ["V10D_DIAGNOSTIC"], "args": os.environ["V10D_ARGS"],
     "status": os.environ["V10D_STATUS"], "exit_code": int(os.environ["V10D_EXIT"]),
     "partition": os.environ.get("SLURM_JOB_PARTITION"), "node": os.uname().nodename,
     "gpu": query.stdout.strip() if query.returncode == 0 else "nvidia-smi rc=%d" % query.returncode,
     "freeze_sha256": os.environ["V10D_FREEZE_SHA256"], "slurm_start": os.environ.get("V10D_SCHED_START"),
     "start_unix": start, "batch_start_unix": float(os.environ["V10D_BATCH_START"]), "end_unix": end,
     "allocated_seconds": end - start, "batch_seconds": end - float(os.environ["V10D_BATCH_START"]),
     "label": "post-DEV sensitivity diagnostic"}
p = Path(sys.argv[1]) / "exits"; p.mkdir(parents=True, exist_ok=True)
t = p / ("." + r["job"] + ".tmp"); t.write_text(json.dumps(r, indent=2) + "\n"); t.replace(p / (r["job"] + ".json"))
PY
}

case " $allowed " in
  *" ${SLURM_JOB_PARTITION:-none} "*) ;;
  *) echo "[v10d] REFUSED: $diag is pre-registered for partition(s) $allowed, got ${SLURM_JOB_PARTITION:-none}" >&2
     receipt WRONG_PARTITION 4; exit 4 ;;
esac

echo "[v10d] nvidia-smi (full):"
nvidia-smi || echo "[v10d] nvidia-smi returned $?"
nvidia-smi --query-gpu=index,name,uuid,driver_version,compute_cap,memory.used,memory.total,utilization.gpu --format=csv || true
set +e
"$PYTHON" - <<'PY'
import sys, torch
try:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    x = torch.ones(1024, device="cuda"); torch.cuda.synchronize(); assert float(x.sum()) == 1024.0
    print("[v10d] CUDA_PROBE_OK", torch.cuda.get_device_name(0), torch.version.cuda, torch.backends.cudnn.version(), flush=True)
except Exception as error:
    print("[v10d] CUDA_PROBE_FAILED:", repr(error), flush=True)
    sys.exit(3)
PY
probe=$?
set -e
if [ "$probe" -ne 0 ]; then
  echo "[v10d] INFRA_CUDA_UNAVAILABLE on $(hostname): the CUDA device could not be initialised; no diagnostic work was started" >&2
  receipt INFRA_CUDA_UNAVAILABLE 3
  exit 3
fi

cd "$source_root"
child=''
forward_signal() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap forward_signal USR1 TERM
"$PYTHON" -u "$DIAG_ROOT/scripts/v10_sensitivity/$script" "$@" &
child=$!
set +e
wait "$child"
rc=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; rc=$?; fi
set -e
if [ "$rc" -eq 0 ]; then receipt COMPLETE 0; else receipt FAIL "$rc"; fi
echo "[v10d] diag=$diag args=$* rc=$rc end=$(date -u +%FT%TZ)"
exit "$rc"
