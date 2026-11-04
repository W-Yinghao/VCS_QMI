#!/bin/bash
# R44 formal submission after CPU_TEST_R44 ALL_PASS: freeze, then one job per cell running J/BCE/NCE; normal nice 50, RTX6000PRO/L40S/H100 excl.
# node51/53/60; self-resume <= 3 (units_r44.sbatch). A (16000 updates, Flickr joint): 64G/16h (as R41-A); B/C Flickr joint 64G/8h; CC joint 150G/10h.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103; cd $W; PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
[ -n "${DRY:-}" ] || [ -f PROTOCOL_FROZEN_R44.json ] || $PY -I -B tools/freeze_r44.py
sub() { local name=$1 tree=$2 mem=$3 tl=$4 units=$5 j
  if [ -z "${DRY:-}" ] && squeue -u yinwang -h -o %j | grep -qx "$name"; then echo "$name already queued"; return; fi
  [ -n "${DRY:-}" ] && { echo "$name $tree $mem $tl $units"; return; }
  j=$(sbatch --parsable --qos=normal --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 --nice=50 --mem=$mem --time=$tl \
      --export=ALL,R40_MEM=$mem,R40_TIME=$tl -J $name sbatch/units_r44.sbatch $tree train_unit.py "$units")
  printf "%s\t%s\t%s\tR44 formal (%s/train_unit.py, mem %s, time %s, nice 50): %s\n" "$(date -u +%FT%TZ)" $name $j $tree $mem $tl "$units" >> SUBMITTED_R44.tsv; echo "$name $j"; }
three() { echo "${1}_J_${2},${1}_BCE_${2},${1}_NCE_${2}"; }
for P in F60_R30_H1001 F60_R30_H1002 F60_R30_H1003; do sub r44_u_A_${P}_16k_s2002 r44f 64G 16:00:00 "$(three R44_A_${P}_ALL train2002)"; done
for C in F00_R30_H1001 F60_R30_H1001 F80_IN2R_H1001; do for A in ROUTE_PRESENT ROUTE_RANDOM; do sub r44_u_B_${C}_${A} r44f 64G 08:00:00 "$(three R44_B_${C}_${A} train1001)"; done; done
for A in ROUTE_PRESENT ROUTE_RANDOM; do sub r44_u_B_CC_R31_H1001_${A} r44c 150G 10:00:00 "$(three R44_B_CC_R31_H1001_${A} train1001)"; done
for C in F00_R30_H1001 F80_IN2R_H1001; do sub r44_u_C_${C}_PRESENT_ALL r44f 64G 08:00:00 "$(three R44_C_${C}_PRESENT_ALL train1001)"; done
