#!/bin/bash
# R35 formal submission (run once, after CPU_TEST + GPU_GATE ALL_PASS): freeze, then 4 bundled GPU jobs at nice 0.
# D units run in the same job directly after their C-L parent reaches step 8000 (no --dependency, no outcome gate).
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r35_expanded_20261102; cd $W
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
TL=${TL:-12:00:00}
[ -f PROTOCOL_FROZEN.json ] || $PY -B r35/freeze.py
squeue -u yinwang -h -o %j | grep -q '^r35_[AC][BD]*_n' && { echo "R35 formal jobs already queued"; exit 1; }
u() { local c=$1; shift; local o=""; for x in "$@"; do o="$o,r35_${x}"; done; echo "${o#,}"; }
A00=$(u n00 A_n00_s1001_ID_J A_n00_s1001_ID_BCE A_n00_s1001_ID_NCE A_n00_s1001_G_J A_n00_s1001_G_BCE A_n00_s1001_G_NCE)
A60=$(u n60 A_n60_s1001_ID_J A_n60_s1001_ID_BCE A_n60_s1001_ID_NCE A_n60_s1001_G_J A_n60_s1001_G_BCE A_n60_s1001_G_NCE B_n60_s1001_TRUST_J B_n60_s1001_TRUST_BCE B_n60_s1001_TRUST_NCE)
OR60=$(u n60 B_n60_s1001_ORACLE_J B_n60_s1001_ORACLE_BCE B_n60_s1001_ORACLE_NCE)
cd_list() { local c=$1; local o=""; for ob in J BCE NCE; do o="$o,r35_C_${c}_s1001_L_${ob},r35_D_${c}_s1001_L_${ob},r35_D_${c}_s1001_L_${ob}_HEAD_CONTROL"; done
  for ob in J BCE NCE; do o="$o,r35_C_${c}_s1001_G_${ob}"; done; echo "${o#,}"; }
CD00=$(cd_list n00); CD60=$(cd_list n60)
sub() { local name=$1; shift; local j; j=$(sbatch --parsable -J $name --time=$TL sbatch/units.sbatch "$@"); printf "%s\t%s\t%s\tformal: %s\n" "$(date -u +%FT%TZ)" $name $j "$*" >> SUBMITTED.tsv; echo "$name $j"; }
sub r35_A_n00 "train_unit:$A00"
sub r35_AB_n60 "train_unit:$A60" "train_oracle:$OR60"
sub r35_CD_n00 "train_unit:$CD00"
sub r35_CD_n60 "train_unit:$CD60"
