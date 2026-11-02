#!/bin/bash
# R39 formal submission (once, after CPU_TEST_pre/post ALL_PASS + both JOINT functional checks): freeze, then 9 normal nice-0 GPU
# jobs = 21 units (oracle units in their own process/dir). Time limits from the measured JOINT seconds/update (JOINT_SMOKE.json).
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r39_complete_joint_20261102; cd $W
[ -n "${DRY:-}" ] || [ -f PROTOCOL_FROZEN.json ] || python3 -I -B tools/freeze.py
if [ -z "${DRY:-}" ] && squeue -u yinwang -h -o %j | grep -q '^r39_u_'; then echo "R39 formal jobs already queued"; exit 1; fi
sub() { local name=$1 tree=$2 runner=$3 mem=$4 tl=$5 units=$6 j
  [ -n "${DRY:-}" ] && { echo "$name $tree $runner $mem $tl $units"; return; }
  j=$(sbatch --parsable --qos=normal --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 --nice=0 --mem=$mem --time=$tl \
      --export=ALL,R39_MEM=$mem,R39_TIME=$tl -J $name sbatch/units.sbatch $tree $runner "$units")
  printf "%s\t%s\t%s\tformal (%s/%s, mem %s, time %s): %s\n" "$(date -u +%FT%TZ)" $name $j $tree $runner $mem $tl "$units" >> SUBMITTED.tsv; echo "$name $j"; }
F=R39_Flickr60; C=R39_CC152K
TJF=${TJF:-08:00:00}; TJC=${TJC:-10:00:00}
sub r39_u_f60_cfrozen   r39f train_unit.py 64G  02:00:00 "${F}_COMPLETE_frozen_J,${F}_COMPLETE_frozen_BCE,${F}_COMPLETE_frozen_NCE"
sub r39_u_cc_cfrozen    r39c train_unit.py 64G  02:00:00 "${C}_COMPLETE_frozen_J,${C}_COMPLETE_frozen_BCE,${C}_COMPLETE_frozen_NCE"
sub r39_u_f60_oracle    r39f train_oracle.py 64G 02:00:00 "${F}_ORACLE_MISSING_frozen_J,${F}_ORACLE_MISSING_frozen_BCE,${F}_ORACLE_MISSING_frozen_NCE"
sub r39_u_cc_joint_J    r39c train_unit.py 150G $TJC "${C}_N_joint_global_J,${C}_COMPLETE_joint_global_J"
sub r39_u_cc_joint_BCE  r39c train_unit.py 150G $TJC "${C}_N_joint_global_BCE,${C}_COMPLETE_joint_global_BCE"
sub r39_u_cc_joint_NCE  r39c train_unit.py 150G $TJC "${C}_N_joint_global_NCE,${C}_COMPLETE_joint_global_NCE"
sub r39_u_f60_joint_J   r39f train_unit.py 64G  $TJF "${F}_N_joint_global_J,${F}_COMPLETE_joint_global_J"
sub r39_u_f60_joint_BCE r39f train_unit.py 64G  $TJF "${F}_N_joint_global_BCE,${F}_COMPLETE_joint_global_BCE"
sub r39_u_f60_joint_NCE r39f train_unit.py 64G  $TJF "${F}_N_joint_global_NCE,${F}_COMPLETE_joint_global_NCE"
