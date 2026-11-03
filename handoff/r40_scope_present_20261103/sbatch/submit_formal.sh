#!/bin/bash
# R40 formal submission (once, after CPU_TEST.json ALL_PASS): freeze, then 9 normal GPU jobs at nice 100 (after R39X nice 0, above R33 nice
# 10000), one job per (dataset, block, adaptation) running the matched J/BCE/NCE units in order. Resources: JOINT jobs as the R39 JOINT jobs
# (Flickr 64G/8h, CC 150G/10h); head-only PRESENT_frozen jobs as the R39 frozen jobs (64G/2h). Self-resume <= 3 as R39. Never runfill.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103; cd $W
[ -n "${DRY:-}" ] || [ -f PROTOCOL_FROZEN.json ] || /projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -I -B tools/freeze.py
if [ -z "${DRY:-}" ] && squeue -u yinwang -h -o %j | grep -q '^r40_u_'; then echo "R40 formal jobs already queued"; exit 1; fi
sub() { local name=$1 tree=$2 runner=$3 mem=$4 tl=$5 units=$6 j
  [ -n "${DRY:-}" ] && { echo "$name $tree $runner $mem $tl $units"; return; }
  j=$(sbatch --parsable --qos=normal --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 --nice=100 --mem=$mem --time=$tl \
      --export=ALL,R40_MEM=$mem,R40_TIME=$tl -J $name sbatch/units.sbatch $tree $runner "$units")
  printf "%s\t%s\t%s\tR40 formal (%s/%s, mem %s, time %s, nice 100): %s\n" "$(date -u +%FT%TZ)" $name $j $tree $runner $mem $tl "$units" >> SUBMITTED.tsv; echo "$name $j"; }
three() { echo "${1}_J,${1}_BCE,${1}_NCE"; }
# block A + C first (package section 6: A and C before B), then B
sub r40_u_F60_A_image_only r40f train_unit.py   64G  08:00:00 "$(three R40_A_F60_R30_H1001_image_only)"
sub r40_u_F60_A_text_only  r40f train_unit.py   64G  08:00:00 "$(three R40_A_F60_R30_H1001_text_only)"
sub r40_u_F60_C_oracle_joint r40f train_oracle.py 64G 08:00:00 "$(three R40_C_F60_R30_H1001_ORACLE_joint)"
sub r40_u_CC_A_image_only  r40c train_unit.py   150G 10:00:00 "$(three R40_A_CC31_H1001_image_only)"
sub r40_u_CC_A_text_only   r40c train_unit.py   150G 10:00:00 "$(three R40_A_CC31_H1001_text_only)"
sub r40_u_F60_B_present_joint r40f train_unit.py 64G 08:00:00 "$(three R40_B_F60_R30_H1001_PRESENT_both_joint)"
sub r40_u_CC_B_present_joint  r40c train_unit.py 150G 10:00:00 "$(three R40_B_CC31_H1001_PRESENT_both_joint)"
sub r40_u_F60_B_present_frozen r40f train_unit.py 64G 02:00:00 "$(three R40_B_F60_R30_H1001_PRESENT_frozen)"
sub r40_u_CC_B_present_frozen  r40c train_unit.py 64G 02:00:00 "$(three R40_B_CC31_H1001_PRESENT_frozen)"
