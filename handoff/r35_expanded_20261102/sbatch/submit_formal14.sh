#!/bin/bash
# R35 formal submission, 14 jobs (user 2026-11-02 「拆成14个，R33的让位，尽快跑」), replacing the 4-bundle submit_formal.sh.
# Run once after GPU_GATE ALL_PASS. Freezes the protocol, then submits normal nice 0 jobs. Each D unit still runs in the same
# job right after its C-L parent (no --dependency, no outcome gate). Same unit ids and runner as submit_formal.sh.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r35_expanded_20261102; cd $W
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
TL=${TL:-12:00:00}
[ -f PROTOCOL_FROZEN.json ] || $PY -B r35/freeze.py
squeue -u yinwang -h -o %j | grep -q '^r35_[ABCD]' && { echo "R35 formal jobs already queued"; exit 1; }
L() { local o=""; for x in "$@"; do o="$o,r35_${x}"; done; echo "${o#,}"; }
sub() { local name=$1; shift; local j; j=$(sbatch --parsable -J $name --time=$TL sbatch/units.sbatch "$@"); printf "%s\t%s\t%s\tformal(14-split): %s\n" "$(date -u +%FT%TZ)" $name $j "$*" >> SUBMITTED.tsv; echo "$name $j"; }
for c in n00 n60; do
  for s in ID G; do sub r35_A_${c}_${s} "train_unit:$(L A_${c}_s1001_${s}_J A_${c}_s1001_${s}_BCE A_${c}_s1001_${s}_NCE)"; done
done
sub r35_B_n60_TRUST "train_unit:$(L B_n60_s1001_TRUST_J B_n60_s1001_TRUST_BCE B_n60_s1001_TRUST_NCE)"
sub r35_B_n60_ORACLE "train_oracle:$(L B_n60_s1001_ORACLE_J B_n60_s1001_ORACLE_BCE B_n60_s1001_ORACLE_NCE)"
for c in n00 n60; do
  for ob in J BCE NCE; do sub r35_CD_${c}_${ob} "train_unit:$(L C_${c}_s1001_L_${ob} D_${c}_s1001_L_${ob} D_${c}_s1001_L_${ob}_HEAD_CONTROL)"; done
  sub r35_C_${c}_G "train_unit:$(L C_${c}_s1001_G_J C_${c}_s1001_G_BCE C_${c}_s1001_G_NCE)"
done
