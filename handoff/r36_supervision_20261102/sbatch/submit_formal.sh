#!/bin/bash
# R36 formal submission (run once after CPU_TEST + GPU_GATE ALL_PASS): freeze the protocol, then 8 normal nice-0 GPU jobs
# covering all 27 registered units (V: both phases inside the unit, no --dependency, no outcome gate).
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r36_supervision_20261102; cd $W
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
[ -f PROTOCOL_FROZEN.json ] || $PY -B r36/freeze.py
squeue -u yinwang -h -o %j | grep -q '^r36_[MTNV]_' && { echo "R36 formal jobs already queued"; exit 1; }
L() { local o=""; for x in "$@"; do o="$o,r36_${x}"; done; echo "${o#,}"; }
sub() { local name=$1; shift; local j; j=$(sbatch --parsable -J $name sbatch/units.sbatch "$@"); printf "%s\t%s\t%s\tformal: %s\n" "$(date -u +%FT%TZ)" $name $j "$*" >> SUBMITTED.tsv; echo "$name $j"; }
sub r36_M_RANDOM "train_unit:$(L M_n60_G_RANDOM_J M_n60_G_RANDOM_BCE M_n60_G_RANDOM_NCE)"
sub r36_M_ORACLE "train_oracle:$(L M_n60_G_SHIELD_ORACLE_J M_n60_G_SHIELD_ORACLE_BCE M_n60_G_SHIELD_ORACLE_NCE M_n60_G_RELABEL_AVAILABLE_ORACLE_J M_n60_G_RELABEL_AVAILABLE_ORACLE_BCE M_n60_G_RELABEL_AVAILABLE_ORACLE_NCE)"
for s in G L; do
  sub r36_T_${s} "train_unit:$(L T_n60_${s}_J T_n60_${s}_BCE T_n60_${s}_NCE)"
  sub r36_N_${s} "train_unit:$(L N_n60_${s}_J N_n60_${s}_BCE N_n60_${s}_NCE)"
done
for h in 2002 2003; do sub r36_V_h${h} "train_unit:$(L V_n00_h${h}_L_J V_n00_h${h}_L_BCE V_n00_h${h}_L_NCE)"; done
