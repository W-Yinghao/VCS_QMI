#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101/logs/%x_%j.out
# R30 v1.1 parent state (5 warmup epochs, source H or L) -> parents/<parent_id>/parent_state.pt (read-only) + PARENT.json.
# Usage: sbatch --partition=<block partition> --exclude=node51,node53,node60 r30v11_parent.sbatch PARENT_ID
# tools/run_r30.py --stage parent --execute (gate PASS + frozen v1.1 protocol + code sha; block GPU enforced by the driver).
set -euo pipefail
PID=$1
source /home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101/server.env
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source "$R27_STAGE_SH"
"$PYTHON_BIN" -u -B $R30_ROOT/tools/run_r30.py --stage parent --parent $PID --root $R30_ROOT --cells $R30_CELLS --code-dir $R30V11_CODE --data-path "$DP" --out-base $R30_OUT_BASE --python "$PYTHON_BIN" --gate-report $R30_GATE --execute
echo R30_PARENT_DONE $PID
