#!/usr/bin/env bash
# V10 TEST_REPEAT134 input diagnostic (CPU only; scope DIAGNOSTIC_AUTHORIZATION_V4D.json):  diag.sbatch <core>
# Numbers only: no evaluation, no perturbation or truth, no registration result.  Never a GPU.
#SBATCH --partition=cpu-high
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset NPY_DISABLE_CPU_FEATURES NPY_ENABLE_CPU_FEATURES
: "${V10_TR_EXPECT_ACTIVATION:?v4 activation sha256 pin required}"
: "${V10_TR_EXPECT_COMMIT:?diagnostic execution commit pin required}"
: "${V10_TR_EXPECT_DIAG:?diagnostic authorization sha256 pin required}"
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_test_repeat_exec_v4diag_20261020
BUNDLE=/home/infres/yinwang/CS_QMI/handoff/v10_test_repeat134_20261020
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
export V10_TR_START
V10_TR_START=$(date +%s)
core=${1:?core required}
set +e
"$PYTHON" -u "$R/scripts/v10_test_repeat/diagnose_inputs.py" "$core"
rc=$?
export V10_TR_EXIT=$rc V10_TR_TARGET=$core
"$PYTHON" - "$BUNDLE" <<'PY'
import json, os, sys, time
from pathlib import Path
p = Path(sys.argv[1]) / "diagnostics" / "exits"; p.mkdir(parents=True, exist_ok=True)
r = {"job": os.environ["SLURM_JOB_ID"], "exit_code": int(os.environ["V10_TR_EXIT"]), "stage": "diagnose",
     "target": os.environ["V10_TR_TARGET"], "activation_sha256": os.environ["V10_TR_EXPECT_ACTIVATION"],
     "diagnostic_authorization_sha256": os.environ["V10_TR_EXPECT_DIAG"], "commit": os.environ["V10_TR_EXPECT_COMMIT"],
     "partition": os.environ.get("SLURM_JOB_PARTITION"), "nodelist": os.environ.get("SLURM_JOB_NODELIST"),
     "start_unix": float(os.environ["V10_TR_START"]), "end_unix": time.time()}
r["allocated_seconds"] = r["end_unix"] - r["start_unix"]
tmp = p / ("." + r["job"] + ".tmp"); tmp.write_text(json.dumps(r, indent=2) + "\n"); tmp.replace(p / (r["job"] + ".json"))
PY
receipt_rc=$?
if [ "$rc" -eq 0 ] && [ "$receipt_rc" -ne 0 ]; then rc=$receipt_rc; fi
exit "$rc"
