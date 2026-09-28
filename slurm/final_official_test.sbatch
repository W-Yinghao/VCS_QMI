#!/usr/bin/env bash
#SBATCH --job-name=final_official_test
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/final_official_test_%x_%j.out
# P67/P68: the one-shot official CIFAR-10 test evaluation.  UNITS (units file) and REPORT (output prefix) required.
# STANDIN=1 -> smoke on the selection split (the test partition is never opened; VCS_FINAL_ROUND is NOT exported).  LIMIT=N -> first N units.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${UNITS:?}"; : "${REPORT:?}"
if [ "${STANDIN:-0}" = "1" ]; then
  banner "P68 STAND-IN smoke (selection split): $UNITS -> $REPORT"
  "$PYTHON" scripts/final_official_test_eval.py --units "$UNITS" --out "$REPORT" --standin ${LIMIT:+--limit $LIMIT} ${EXTRA_ARGS:-}; rc=$?
else
  banner "P68 OFFICIAL TEST evaluation (final round): $UNITS -> $REPORT"
  export VCS_FINAL_ROUND=1
  "$PYTHON" scripts/final_official_test_eval.py --units "$UNITS" --out "$REPORT" ${LIMIT:+--limit $LIMIT} ${EXTRA_ARGS:-}; rc=$?
fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
