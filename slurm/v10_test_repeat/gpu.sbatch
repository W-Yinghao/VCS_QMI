#!/usr/bin/env bash
# V10 TEST_REPEAT134 stages:  gate <gate_id> | prepare | test <unit_key>
# --job-name, --partition and --time are given per job on the sbatch line (see the protocol job plan).
#SBATCH --qos=normal
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
: "${V10_TR_EXPECT_ACTIVATION:?activation sha256 pin required at submission}"
: "${V10_TR_EXPECT_COMMIT:?execution commit pin required at submission}"
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_test_repeat_exec_v4_20261020
BUNDLE=/home/infres/yinwang/CS_QMI/handoff/v10_test_repeat134_20261020
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
export V10_TR_START
V10_TR_START=$(date +%s)
mode=${1:?gate|prepare|test}
target=${2:-}
case "$mode" in
  gate|test) script=evaluate.py; set -- "$mode" "${target:?unit or gate id required}" ;;
  prepare) script=prepare_inputs.py; set -- ;;
  *) echo "unknown stage $mode" >&2; exit 2 ;;
esac
child=''
forward_signal() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi; }
trap forward_signal USR1 TERM
"$PYTHON" -u "$R/scripts/v10_test_repeat/$script" "$@" &
child=$!
set +e
wait "$child"
rc=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; rc=$?; fi
export V10_TR_EXIT=$rc V10_TR_MODE=$mode V10_TR_TARGET=$target
"$PYTHON" - "$BUNDLE" <<'PY'
import json, os, sys, time
from pathlib import Path
b = Path(sys.argv[1]); p = b / "exits"; p.mkdir(parents=True, exist_ok=True)
r = {"job": os.environ["SLURM_JOB_ID"], "exit_code": int(os.environ["V10_TR_EXIT"]), "stage": os.environ["V10_TR_MODE"],
     "target": os.environ["V10_TR_TARGET"], "activation_sha256": os.environ["V10_TR_EXPECT_ACTIVATION"],
     "commit": os.environ["V10_TR_EXPECT_COMMIT"], "partition": os.environ.get("SLURM_JOB_PARTITION"),
     "nodelist": os.environ.get("SLURM_JOB_NODELIST"), "start_unix": float(os.environ["V10_TR_START"]), "end_unix": time.time()}
r["allocated_seconds"] = r["end_unix"] - r["start_unix"]
tmp = p / ("." + r["job"] + ".tmp"); tmp.write_text(json.dumps(r, indent=2) + "\n"); tmp.replace(p / (r["job"] + ".json"))
PY
receipt_rc=$?
if [ "$rc" -eq 0 ] && [ "$receipt_rc" -ne 0 ]; then rc=$receipt_rc; fi
exit "$rc"
