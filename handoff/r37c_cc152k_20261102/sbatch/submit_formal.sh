#!/bin/bash
# R37C formal submission (once, after CPU_TEST ALL_PASS): freeze, then 2 normal nice-0 GPU jobs = 6 units.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r37c_cc152k_20261102; cd $W
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
[ -f PROTOCOL_FROZEN.json ] || $PY -B r37c/freeze.py
squeue -u yinwang -h -o %j | grep -q '^r37c_cc_[TN]$' && { echo "R37C formal jobs already queued"; exit 1; }
sub() { local name=$1 units=$2 j; j=$(sbatch --parsable -J $name sbatch/units.sbatch "$units"); printf "%s\t%s\t%s\tformal: %s\n" "$(date -u +%FT%TZ)" $name $j "$units" >> SUBMITTED.tsv; echo "$name $j"; }
for r in T N; do sub r37c_cc_${r} "r37c_pilot_cc_h1001_${r}_G_J,r37c_pilot_cc_h1001_${r}_G_BCE,r37c_pilot_cc_h1001_${r}_G_NCE"; done
