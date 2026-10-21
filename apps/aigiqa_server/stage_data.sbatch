#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# CPU staging after the coordinator's data_fetch download of group "aigiqa": verify fetch records and pinned trees,
# owner-MOS mapping check (counts only), image validation + HF-mirror hash cross-check, offline tokenizer cache,
# the package's `aigiqa.cli prepare` (prompt groups 70/15/15, seed 20261019), TEST seal; then the server base config.
set -uo pipefail
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_aigiqa/apps/aigiqa_server/common.sh
"$AIG_PY" "$AIG_SERVER/freeze_env.py" check --lock "$AIG_SERVER/ENV_LOCK.json" || exit $?
"$AIG_PY" "$AIG_SERVER/stage_data.py" --data "$AIG_DATA" --models "$AIG_MODELS" --hf-home "$AIG_HF_HOME" \
  --bert-local "$AIG_BERT_LOCAL" --pkg "$AIG_PKG" --manifest "$AIG_REPO/data/DATA_MANIFEST.json" \
  --evidence "$AIG_SERVER/evidence/staging" || exit $?
"$AIG_PY" "$AIG_SERVER/make_matrix.py" base
