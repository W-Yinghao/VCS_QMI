#!/bin/bash
# R46 formal submission after CPU_TEST_R46 ALL_PASS and both GPU smokes ALL_PASS: freeze (args: smoke job ids), check_frozen, then one job per
# cell running J/BCE/NCE; normal nice 50; self-resume <= 3 (units_r46.sbatch, which resubmits on RTX6000PRO,L40S,H100 unless PIN set).
# A (COCO pools): frozen 64G/4h, joint 150G/12h on RTX6000PRO,L40S,H100. B: pinned to the round-1 GPU type (Flickr L40S 64G/8h, COCO RTX6000PRO 150G/12h).
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103; cd $W; PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
[ -f PROTOCOL_FROZEN_R46.json ] || $PY -I -B tools/freeze_r46.py $1 $2
for t in r46f r46o; do (cd /tmp && $PY -I -B -c "
import sys,json; W='$W'; sys.path.insert(0,W+'/$t'); import train_unit as TU
u=[x for x in json.load(open(W+'/configs/UNITS_R46.json')) if json.load(open(f'{W}/configs/{x}.json'))['code_tree']=='$t']; TU.check_frozen(u); print('$t check_frozen OK', len(u))"); done
sub() { local name=$1 tree=$2 mem=$3 tl=$4 part=$5 units=$6 j
  if squeue -u yinwang -h -o %j | grep -qx "$name"; then echo "$name already queued"; return; fi
  j=$(sbatch --parsable --qos=normal --partition=$part --exclude=node51,node53,node60 --nice=50 --mem=$mem --time=$tl \
      --export=ALL,R40_MEM=$mem,R40_TIME=$tl,R46_PART=$part -J $name sbatch/units_r46.sbatch $tree train_unit.py "$units")
  printf "%s\t%s\t%s\tR46 formal (%s/train_unit.py, mem %s, time %s, partition %s, nice 50): %s\n" "$(date -u +%FT%TZ)" $name $j $tree $mem $tl $part "$units" >> SUBMITTED_R46.tsv; echo "$name $j"; }
three() { echo "${1}_J,${1}_BCE,${1}_NCE"; }
ANY=RTX6000PRO,L40S,H100
for N in 29 58; do
  sub r46_u_A_P${N}K_joint r46o 150G 12:00:00 $ANY "$(three R46_A_C80_IN2R_H1001_P${N}K_N_joint)"
  sub r46_u_A_P${N}K_frozen r46o 64G 04:00:00 $ANY "$(three R46_A_C80_IN2R_H1001_P${N}K_N_frozen)"; done
for A in REBUILT CONTINUED; do
  sub r46_u_B_F60_${A} r46f 64G 08:00:00 L40S "$(three R46_B_F60_R30_H1001_R2_${A})"
  sub r46_u_B_F80_${A} r46f 64G 08:00:00 L40S "$(three R46_B_F80_IN2R_H1001_R2_${A})"
  sub r46_u_B_C80_${A} r46o 150G 12:00:00 RTX6000PRO "$(three R46_B_C80_IN2R_H1001_R2_${A})"; done
