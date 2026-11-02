#!/bin/bash
# R37 formal submission (once, after CPU_TEST + GPU_GATE ALL_PASS): freeze, then 6 normal nice-0 GPU jobs = 18 units.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r37_confirm_20261102; cd $W
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
[ -f PROTOCOL_FROZEN.json ] || $PY -B r37/freeze.py
squeue -u yinwang -h -o %j | grep -q '^r37_h100[123]_' && { echo "R37 formal jobs already queued"; exit 1; }
sub() { local name=$1 units=$2 j; j=$(sbatch --parsable -J $name sbatch/units.sbatch "$units"); printf "%s\t%s\t%s\tformal: %s\n" "$(date -u +%FT%TZ)" $name $j "$units" >> SUBMITTED.tsv; echo "$name $j"; }
for h in 1002 1003; do for r in T N; do
  sub r37_h${h}_${r} "r37_confirm_f60_h${h}_${r}_G_J,r37_confirm_f60_h${h}_${r}_G_BCE,r37_confirm_f60_h${h}_${r}_G_NCE"
done; done
sub r37_h1001_TN "r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_J,r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_BCE,r37_factor_f60_h1001_TRUST_NATIVE_NULL_G_NCE"
sub r37_h1001_PO "r37_factor_f60_h1001_N_PRESENT_ONLY_G_J,r37_factor_f60_h1001_N_PRESENT_ONLY_G_BCE,r37_factor_f60_h1001_N_PRESENT_ONLY_G_NCE"
