#!/bin/bash
#SBATCH --job-name=r21_aig_chk_cputests
#SBATCH --qos=normal
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --nice=0
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Package CPU tests (tests/test_application_update.py) on the installed stage, once in each app's env.
set -uo pipefail
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_round21/apps/round_20261021/server/paths.env
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONPATH
OUT=$R21_WT/apps/round_20261021/records/cpu_tests; mkdir -p "$OUT"
cd "$OUT" || exit 2
rc_all=0
for pair in "aigenv:$AIG_PY" "kdenv:$KD_PY"; do
  tag=${pair%%:*}; py=${pair#*:}
  t0=$(date +%s)
  "$py" "$R21_APPS/tests/test_application_update.py" --package-root "$R21_STAGE" --report "$OUT/APPS_CPU_TESTS_${tag}.json"
  rc=$?; echo "$tag rc=$rc seconds=$(( $(date +%s) - t0 )) node=$(hostname)"; [ $rc -eq 0 ] || rc_all=$rc
done
exit $rc_all
