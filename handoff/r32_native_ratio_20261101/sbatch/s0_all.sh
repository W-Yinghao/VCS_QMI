#!/bin/bash
# NRCS-v1 S0: all 27 registered synthetic rows via the package's tools/run_registered.py (exact population, CPU)
set -uo pipefail
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r32_native_ratio_20261101
cd $W/package
fail=0
for U in $(awk -F, '$2=="S0"{print $1}' configs/MATRIX.csv); do
  if [ -f $O/runs/$U/execution_receipt.json ]; then echo "skip done $U"; continue; fi
  t0=$(date +%s)
  if $PY -B tools/run_registered.py --unit-id $U --runs-root $O/runs > $O/runs/.$U.log 2>&1; then st=COMPLETE; else st=FAILED_TECHNICAL; fail=$((fail+1)); fi
  printf "%s\t%s\t%s\t%ss\t%s\n" "$(date -u +%FT%TZ)" "$U" "$st" "$(( $(date +%s)-t0 ))" "$(hostname)" >> $O/runs/S0_ATTEMPTS.tsv
done
echo "S0 failures: $fail"
