#!/bin/bash
#SBATCH --job-name=r22_chk_cputests
#SBATCH --qos=normal
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --nice=0
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r22 CPU tests (tests/test_r22.py) on the code snapshot, once in each app env (KD env, AIG env).
set -uo pipefail
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_r22/apps/r22/server/common.sh
CODE=$(readlink -f "$R22_CODE_LINK"); OUT=$R22_WT/apps/r22/records/cpu_tests; mkdir -p "$OUT"; cd "$OUT" || exit 2
rc_all=0
for pair in "kdenv:$KD_PY" "aigenv:$AIG_PY"; do
  tag=${pair%%:*}; py=${pair#*:}; t0=$(date +%s)
  "$py" -I "$CODE/tests/test_r22.py" --report "$OUT/R22_CPU_TESTS_${tag}.json"
  rc=$?; echo "$tag rc=$rc seconds=$(( $(date +%s) - t0 )) node=$(hostname) code=$CODE"; [ $rc -eq 0 ] || rc_all=$rc
done
exit $rc_all
