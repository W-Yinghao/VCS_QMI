#!/usr/bin/env bash
#SBATCH --job-name=p137_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P137 gate: core / P130 / P133 test suites (incl. config-hash invariance over all configs) after the v7_head_width marker.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
rc=0; "$PYTHON" -m pytest -q tests/test_ssl_core.py tests/test_vcs_ssl_p130.py tests/test_vcs_ssl_p133.py tests/test_vcs_ssl_p114.py 2>&1 | tail -5 || rc=1
"$PYTHON" - <<'PY' || rc=1
import sys, json; sys.path.insert(0, "src")
from vcs_ssl.config import load_config
for n in json.load(open("configs/P137_SHA256.json"))["configs"]: load_config("configs/" + n)
print("P137 configs load")
PY
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
