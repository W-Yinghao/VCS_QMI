#!/bin/bash
# R43 formal submission after CPU_TEST_R43_<part> ALL_PASS: freeze, then one job per (parent, cell) running J/BCE/NCE; normal nice 50,
# RTX6000PRO/L40S/H100 excl. node51/53/60; self-resume <= 3 (units_r43.sbatch). Part B: Flickr f80 frozen 64G/2h, joint 64G/8h; CC31 frozen 64G/2h,
# joint 150G/10h (R39 CC JOINT setting). Part A (COCO): frozen ${COCO_FROZEN_MEM:-64G}/${COCO_FROZEN_TIME:-06:00:00}, joint
# ${COCO_JOINT_MEM:-128G}/${COCO_JOINT_TIME:-12:00:00} (values from the in-build timing run, registered in R43_ADAPTATION.json).
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103; cd $W; PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
PART=$1; F=$([ $PART = B ] && echo PROTOCOL_FROZEN_R43.json || echo PROTOCOL_FROZEN_R43A.json)
[ -n "${DRY:-}" ] || [ -f $F ] || $PY -I -B tools/freeze_r43.py $PART
sub() { local name=$1 tree=$2 mem=$3 tl=$4 units=$5 j
  if [ -z "${DRY:-}" ] && squeue -u yinwang -h -o %j | grep -qx "$name"; then echo "$name already queued"; return; fi
  [ -n "${DRY:-}" ] && { echo "$name $tree $mem $tl $units"; return; }
  j=$(sbatch --parsable --qos=normal --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 --nice=50 --mem=$mem --time=$tl \
      --export=ALL,R40_MEM=$mem,R40_TIME=$tl -J $name sbatch/units_r43.sbatch $tree train_unit.py "$units")
  printf "%s\t%s\t%s\tR43 formal (%s/train_unit.py, mem %s, time %s, nice 50): %s\n" "$(date -u +%FT%TZ)" $name $j $tree $mem $tl "$units" >> SUBMITTED_R43.tsv; echo "$name $j"; }
three() { echo "${1}_J,${1}_BCE,${1}_NCE"; }
if [ $PART = B ]; then
  for P in F80_IN2R_H1001 F80_IN2R_H1002 F80_IN2R_H1003; do
    sub r43_u_B_${P}_joint r43f 64G 08:00:00 "$(three R43_B_${P}_TRUST_joint)"
    sub r43_u_B_${P}_frozen r43f 64G 02:00:00 "$(three R43_B_${P}_TRUST_frozen)"; done
  sub r43_u_B_CC31_H1001_joint r43c 150G 10:00:00 "$(three R43_B_CC31_H1001_TRUST_joint)"
  sub r43_u_B_CC31_H1001_frozen r43c 64G 02:00:00 "$(three R43_B_CC31_H1001_TRUST_frozen)"
else
  for P in C60_IN2R_H1001 C80_IN2R_H1001 C20_IN2R_H1001 C60_IN2R_H1002 C60_IN2R_H1003 C80_IN2R_H1002 C80_IN2R_H1003; do
    sub r43_u_A_${P}_joint r43o ${COCO_JOINT_MEM:-128G} ${COCO_JOINT_TIME:-12:00:00} "$(three R43_A_${P}_N_joint)"
    sub r43_u_A_${P}_frozen r43o ${COCO_FROZEN_MEM:-64G} ${COCO_FROZEN_TIME:-06:00:00} "$(three R43_A_${P}_N_frozen)"; done
fi
