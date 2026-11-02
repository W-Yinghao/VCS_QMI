#!/bin/bash
# R39X formal submission (once, after CPU_TEST_R39X ALL_PASS + 5 JOINT preps): freeze, then 6 normal nice-0 GPU jobs (one per parent;
# 3 units each, the 0% parent 3 frozen + 3 JOINT), same mem/time as the R39 JOINT jobs (Flickr 64G/8h, CC 150G/10h), self-resume as R39.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102; cd $W
[ -n "${DRY:-}" ] || [ -f PROTOCOL_FROZEN_R39X.json ] || python3 -I -B tools/freeze_x.py
if [ -z "${DRY:-}" ] && squeue -u yinwang -h -o %j | grep -q '^r39x_u_'; then echo "R39X formal jobs already queued"; exit 1; fi
sub() { local name=$1 tree=$2 mem=$3 tl=$4 units=$5 j
  [ -n "${DRY:-}" ] && { echo "$name $tree $mem $tl $units"; return; }
  j=$(sbatch --parsable --qos=normal --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 --nice=0 --mem=$mem --time=$tl \
      --export=ALL,R39_MEM=$mem,R39_TIME=$tl -J $name sbatch/units.sbatch $tree train_unit.py "$units")
  printf "%s\t%s\t%s\tR39X formal (%s, mem %s, time %s): %s\n" "$(date -u +%FT%TZ)" $name $j $tree $mem $tl "$units" >> SUBMITTED.tsv; echo "$name $j"; }
for P in CC152K_R37S_s1001 CC152K_R31H0_s1002 CC152K_R31H0_s1003; do U=R39X_${P}_N_joint_global
  sub r39x_u_$P r39xc 150G 10:00:00 "${U}_J,${U}_BCE,${U}_NCE"; done
U=R39X_Flickr00_R30H_s1001_N
sub r39x_u_Flickr00_R30H_s1001 r39xf 64G 08:00:00 "${U}_frozen_shared_J,${U}_frozen_shared_BCE,${U}_frozen_shared_NCE,${U}_joint_global_J,${U}_joint_global_BCE,${U}_joint_global_NCE"
for P in Flickr60_R30H_s1002 Flickr60_R30H_s1003; do U=R39X_${P}_N_joint_global
  sub r39x_u_$P r39xf 64G 08:00:00 "${U}_J,${U}_BCE,${U}_NCE"; done
