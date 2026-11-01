#!/bin/bash
# NRCS-v1 F1: the 4 registered F1 rows of one noise/seed condition via the package runner (CPU), per-condition binding.
set -uo pipefail
C=$1; O=/projects/EEG-foundation-model/yinghao/outputs_v9/r32_native_ratio_20261101
N=${C%%_*}; N=$((10#${N#n})); S=${C##*_s}
cd $W/package
for U in $(awk -F, -v n=$N -v s=$S '$2=="F1" && $5==n && $4==s {print $1}' configs/MATRIX.csv); do
  [ -f $O/runs/$U/execution_receipt.json ] && { echo "skip $U"; continue; }
  t0=$(date +%s)
  if $PY -B tools/run_registered.py --unit-id $U --binding $O/bindings/BINDING_$C.json > $O/runs/.$U.log 2>&1; then st=COMPLETE; else st=FAILED_TECHNICAL; fi
  printf "%s\t%s\t%s\t%ss\t%s\t%s\n" "$(date -u +%FT%TZ)" "$U" "$st" "$(( $(date +%s)-t0 ))" "$(hostname)" "$SLURM_JOB_ID" >> $O/runs/F1_ATTEMPTS.tsv
done
