#!/usr/bin/env bash
# V10 TEST_REPEAT134 v5 input stage (CPU only):  cpu.sbatch inputs
# The frozen v5 protocol names the variant: A re-prepares under the pinned numpy dispatch into a new root, B1 locks the v4
# rebuild with its accepted exception.  Either writes INPUT_LOCK_V5.json once.  --job-name, --partition (and for A
# --nodelist) are given on the sbatch line (see the protocol inputs.job).  Never a GPU.
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
unset NPY_DISABLE_CPU_FEATURES NPY_ENABLE_CPU_FEATURES
: "${V10_TR_EXPECT_ACTIVATION:?v5 activation sha256 pin required}"
: "${V10_TR_EXPECT_COMMIT:?v5 execution commit pin required}"
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_test_repeat_exec_v5_20261020
BUNDLE=/home/infres/yinwang/CS_QMI/handoff/v10_test_repeat134_20261020
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
export V10_TR_START
V10_TR_START=$(date +%s)
mode=${1:?inputs}
[ "$mode" = inputs ] || { echo "unknown stage $mode" >&2; exit 2; }
set +e
"$PYTHON" -u "$R/scripts/v10_test_repeat/prepare_inputs.py"
rc=$?
export V10_TR_EXIT=$rc
"$PYTHON" - "$BUNDLE" <<'PY'
import json, os, sys, time
from pathlib import Path
p = Path(sys.argv[1]) / "exits"; p.mkdir(parents=True, exist_ok=True)
r = {"job": os.environ["SLURM_JOB_ID"], "exit_code": int(os.environ["V10_TR_EXIT"]), "stage": "inputs_v5", "target": "",
     "activation_sha256": os.environ["V10_TR_EXPECT_ACTIVATION"], "commit": os.environ["V10_TR_EXPECT_COMMIT"],
     "partition": os.environ.get("SLURM_JOB_PARTITION"), "nodelist": os.environ.get("SLURM_JOB_NODELIST"),
     "start_unix": float(os.environ["V10_TR_START"]), "end_unix": time.time()}
r["allocated_seconds"] = r["end_unix"] - r["start_unix"]
tmp = p / ("." + r["job"] + ".tmp"); tmp.write_text(json.dumps(r, indent=2) + "\n"); tmp.replace(p / (r["job"] + ".json"))
PY
receipt_rc=$?
if [ "$rc" -eq 0 ] && [ "$receipt_rc" -ne 0 ]; then rc=$receipt_rc; fi
exit "$rc"
