#!/bin/bash
# R41 formal submission. Part 1 (A/B/D/E) after CPU_TEST_R41_1 ALL_PASS; part 2 (C) after the new-parent prep + CPU_TEST_R41_2 ALL_PASS.
# normal QOS nice 50, RTX6000PRO/L40S/H100 excl. node51/53/60, one job per cell running the matched J/BCE/NCE units; 16000-update jobs get double
# time (Flickr JOINT 16h); head-only frozen jobs 64G/2h as R39/R40. Self-resume <= 3 (units_r41.sbatch). Never runfill.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103; cd $W
PART=$1; PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
case $PART in 1) F=PROTOCOL_FROZEN_R41.json; FT="tools/freeze_r41.py 1"; TREE=r41f;; 2) F=PROTOCOL_FROZEN_R41C.json; FT="tools/freeze_r41.py 2"; TREE=r41f;;
  F) F=PROTOCOL_FROZEN_R41F.json; FT=tools/freeze_r41f.py; TREE=r41c;; G) F=PROTOCOL_FROZEN_R41G.json; FT=tools/freeze_r41g.py; TREE=r41g;; esac
[ -n "${DRY:-}" ] || [ -f $F ] || $PY -I -B $FT
sub() { local name=$1 runner=$2 mem=$3 tl=$4 units=$5 j
  if [ -z "${DRY:-}" ] && squeue -u yinwang -h -o %j | grep -qx "$name"; then echo "$name already queued"; return; fi
  [ -n "${DRY:-}" ] && { echo "$name $runner $mem $tl $units"; return; }
  j=$(sbatch --parsable --qos=normal --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 --nice=50 --mem=$mem --time=$tl \
      --export=ALL,R40_MEM=$mem,R40_TIME=$tl -J $name sbatch/units_r41.sbatch $TREE $runner "$units")
  printf "%s\t%s\t%s\tR41 formal part %s (%s/%s, mem %s, time %s, nice 50): %s\n" "$(date -u +%FT%TZ)" $name $j $PART $TREE $runner $mem $tl "$units" >> SUBMITTED_R41.tsv; echo "$name $j"; }
three() { echo "${1}_J,${1}_BCE,${1}_NCE"; }
if [ $PART = 1 ]; then
  for P in F60_R30_H1001 F60_R30_H1002 F60_R30_H1003 F00_R30_H1001; do sub r41_u_A_${P}_16k train_unit.py 64G 16:00:00 "$(three R41_A_${P}_N_joint_16k)"; done
  sub r41_u_D_F60_R30_H1002_oracle_joint train_oracle.py 64G 08:00:00 "$(three R41_D_F60_R30_H1002_ORACLE_joint)"
  sub r41_u_D_F60_R30_H1002_oracle_frozen train_oracle.py 64G 02:00:00 "$(three R41_D_F60_R30_H1002_ORACLE_frozen)"
  for P in F60_R30_H1002 F60_R30_H1003; do sub r41_u_E_${P}_image_only train_unit.py 64G 08:00:00 "$(three R41_E_${P}_image_only)"; done
  sub r41_u_B_F60_R30_H1001_seed2002 train_unit.py 64G 08:00:00 "$(three R41_B_F60_R30_H1001_seed2002_N_joint)"
elif [ $PART = F ]; then
  sub r41_u_F_CC31_H1001_enclr0p1 train_unit.py 150G 10:00:00 "$(three R41_F_CC31_H1001_N_joint_enclr0p1)"
elif [ $PART = G ]; then
  sub r41_u_G1_F00_R30_H1001_seed2002 train_unit.py 64G 08:00:00 "$(three R41_G1_F00_R30_H1001_seed2002_N_joint)"
  sub r41_u_G2_F60_R30_H1003_oracle_joint train_oracle.py 64G 08:00:00 "$(three R41_G2_F60_R30_H1003_ORACLE_joint)"
  sub r41_u_G2_F60_R30_H1003_oracle_frozen train_oracle.py 64G 02:00:00 "$(three R41_G2_F60_R30_H1003_ORACLE_frozen)"
else
  for P in F40_IN2R_H1001 F80_IN2R_H1001; do
    sub r41_u_C_${P}_joint train_unit.py 64G 08:00:00 "$(three R41_C_${P}_N_joint)"
    sub r41_u_C_${P}_frozen train_unit.py 64G 02:00:00 "$(three R41_C_${P}_N_frozen)"; done
fi
