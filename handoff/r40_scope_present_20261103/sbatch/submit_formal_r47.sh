#!/bin/bash
# R47 formal submission after CPU_TEST_R47 ALL_PASS and the three GPU smokes ALL_PASS: freeze (args: smoke job ids f o c), check_frozen, then one job
# per (cond) cell running J/BCE/NCE; normal nice 50; RTX6000PRO,L40S,H100 (as the source R39 / R39X / R41-C / R43-A JOINT units); self-resume <= 3.
# Flickr 64G/8h, COCO 128G/12h, CC 150G/10h (source-unit resources).
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103; cd $W; PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
[ -f PROTOCOL_FROZEN_R47.json ] || $PY -I -B tools/freeze_r47.py $1 $2 $3
for t in r47f r47o r47c; do (cd /tmp && $PY -I -B -c "
import sys,json; W='$W'; sys.path.insert(0,W+'/$t'); import train_unit as TU
u=[x for x in json.load(open(W+'/configs/UNITS_R47.json')) if json.load(open(f'{W}/configs/{x}.json'))['code_tree']=='$t']; TU.check_frozen(u); print('$t check_frozen OK', len(u))"); done
ANY=RTX6000PRO,L40S,H100
sub() { local name=$1 tree=$2 mem=$3 tl=$4 units=$5 j
  if squeue -u yinwang -h -o %j | grep -qx "$name"; then echo "$name already queued"; return; fi
  if grep -qP "\t$name\t[0-9]+\tR47 formal" SUBMITTED_R47.tsv; then echo "$name already submitted"; return; fi
  until j=$(sbatch --parsable --qos=normal --partition=$ANY --exclude=node51,node53,node60 --nice=50 --mem=$mem --time=$tl \
      --export=ALL,R40_MEM=$mem,R40_TIME=$tl,R47_PART=$ANY -J $name sbatch/units_r47.sbatch $tree train_unit.py "$units" 2>/dev/null); do sleep 60; done
  printf "%s\t%s\t%s\tR47 formal (%s/train_unit.py, mem %s, time %s, partition %s, nice 50): %s\n" "$(date -u +%FT%TZ)" $name $j $tree $mem $tl $ANY "$units" >> SUBMITTED_R47.tsv; echo "$name $j"; }
three() { echo "${1}_J,${1}_BCE,${1}_NCE"; }
for V in G S; do
  sub r47_u_${V}_F60 r47f 64G 08:00:00 "$(three R47_${V}_F60_R30_H1001_N_joint)"
  sub r47_u_${V}_F80 r47f 64G 08:00:00 "$(three R47_${V}_F80_IN2R_H1001_N_joint)"
  sub r47_u_${V}_F00 r47f 64G 08:00:00 "$(three R47_${V}_F00_R30_H1001_N_joint)"
  sub r47_u_${V}_C60 r47o 128G 12:00:00 "$(three R47_${V}_C60_IN2R_H1001_N_joint)"
  sub r47_u_${V}_CC r47c 150G 10:00:00 "$(three R47_${V}_CC31_H1001_N_joint)"; done
