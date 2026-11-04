#!/bin/bash
# R48 formal submission after CPU_TEST_R48 ALL_PASS and the three GPU smokes ALL_PASS: freeze (args: smoke job ids f o c), check_frozen, then one job
# per condition running J/BCE/NCE; normal nice 50; RTX6000PRO,L40S,H100; self-resume <= 3. Flickr 64G/3h, COCO 64G/7h, CC 64G/3h (frozen-unit precedents R41-C / R43-A / R43-B).
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103; cd $W; PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
[ -f PROTOCOL_FROZEN_R48.json ] || $PY -I -B tools/freeze_r48.py $1 $2 $3
for t in r48f r48o r48c; do (cd /tmp && $PY -I -B -c "
import sys,json; W='$W'; sys.path.insert(0,W+'/$t'); import train_unit as TU
u=[x for x in json.load(open(W+'/configs/UNITS_R48.json')) if json.load(open(f'{W}/configs/{x}.json'))['code_tree']=='$t']; TU.check_frozen(u); print('$t check_frozen OK', len(u))"); done
ANY=RTX6000PRO,L40S,H100
sub() { local name=$1 tree=$2 mem=$3 tl=$4 units=$5 j
  if squeue -u yinwang -h -o %j | grep -qx "$name"; then echo "$name already queued"; return; fi
  if grep -qP "\t$name\t[0-9]+\tR48 formal" SUBMITTED_R48.tsv; then echo "$name already submitted"; return; fi
  until j=$(sbatch --parsable --qos=normal --partition=$ANY --exclude=node51,node53,node60 --nice=50 --mem=$mem --time=$tl \
      --export=ALL,R40_MEM=$mem,R40_TIME=$tl,R48_PART=$ANY -J $name sbatch/units_r48.sbatch $tree train_unit.py "$units" 2>/dev/null); do sleep 60; done
  printf "%s\t%s\t%s\tR48 formal (%s/train_unit.py, mem %s, time %s, partition %s, nice 50): %s\n" "$(date -u +%FT%TZ)" $name $j $tree $mem $tl $ANY "$units" >> SUBMITTED_R48.tsv; echo "$name $j"; }
three() { echo "${1}_J,${1}_BCE,${1}_NCE"; }
sub r48_u_F60 r48f 64G 03:00:00 "$(three R48_F60_R30_H1001_N_frozen_lfG)"
sub r48_u_F80 r48f 64G 03:00:00 "$(three R48_F80_IN2R_H1001_N_frozen_lfG)"
sub r48_u_F00 r48f 64G 03:00:00 "$(three R48_F00_R30_H1001_N_frozen_lfG)"
sub r48_u_C60 r48o 64G 07:00:00 "$(three R48_C60_IN2R_H1001_N_frozen_lfG)"
sub r48_u_CC r48c 64G 03:00:00 "$(three R48_CC31_H1001_N_frozen_lfG)"
