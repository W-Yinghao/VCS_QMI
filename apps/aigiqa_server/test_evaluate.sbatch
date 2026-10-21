#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# One-time TEST unseal + package evaluate/compare for the AGIQA-3K alignment pilot (owner 2026-10-21 「开始 (Recommended)」).
# Prerequisite: test_predict.sbatch exited 0 (results/test/PREDICTIONS_DONE.json). Refuses to run a second time.
set -uo pipefail
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_aigiqa/apps/aigiqa_server/common.sh
"$AIG_PY" "$AIG_SERVER/freeze_env.py" check --lock "$AIG_SERVER/ENV_LOCK.json" > /dev/null 2>&1 || exit $?
"$AIG_PY" "$AIG_SERVER/test_evaluate.py" --test-dir "$AIG_OUT/results/test" --prepared "$AIG_PREPARED" --pkg "$AIG_PKG" \
  --evidence "$AIG_SERVER/evidence/test"
