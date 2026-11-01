#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101/logs/%x_%j.out
# R30 v1.1 static calibration on a matching H parent of s1001 (64 labeled FIT batches per peer at its first GMM split)
# -> calib/<parent_id>/COEFFICIENTS_<parent_id>.json. Usage: r30v11_calib.sbatch PARENT_ID (runs once; the runner refuses a rerun).
set -euo pipefail
PID=$1
source /home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101/server.env
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source "$R27_STAGE_SH"
"$PYTHON_BIN" -u -B $R30_ROOT/tools/run_r30.py --stage calib --parent $PID --root $R30_ROOT --cells $R30_CELLS --code-dir $R30V11_CODE --data-path "$DP" --out-base $R30_OUT_BASE --python "$PYTHON_BIN" --gate-report $R30_GATE --bindings $R30_BINDINGS --execute
echo R30_CALIB_DONE $PID
