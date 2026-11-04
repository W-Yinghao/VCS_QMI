#!/bin/bash
# R49 formal submission after CPU_TEST_R49 ALL_PASS and the three GPU smokes ALL_PASS: freeze (args: smoke job ids f o c), check_frozen, then one job
# per code tree running its units in order (frozen units ~1 min each); normal nice 50; RTX6000PRO,L40S,H100; self-resume <= 3.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103; cd $W; PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
[ -f PROTOCOL_FROZEN_R49.json ] || $PY -I -B tools/freeze_r49.py $1 $2 $3
for t in r49f r49o r49c; do (cd /tmp && $PY -I -B -c "
import sys,json; W='$W'; sys.path.insert(0,W+'/$t'); import train_unit as TU
u=[x for x in json.load(open(W+'/configs/UNITS_R49.json')) if json.load(open(f'{W}/configs/{x}.json'))['code_tree']=='$t']; TU.check_frozen(u); print('$t check_frozen OK', len(u))"); done
ANY=RTX6000PRO,L40S,H100
units_of() { $PY -I -B -c "
import json; W='$W'; print(','.join(x for x in json.load(open(W+'/configs/UNITS_R49.json')) if json.load(open(f'{W}/configs/{x}.json'))['code_tree']=='$1'))"; }
sub() { local name=$1 tree=$2 mem=$3 tl=$4 units j; units=$(units_of $tree)
  if squeue -u yinwang -h -o %j | grep -qx "$name"; then echo "$name already queued"; return; fi
  if grep -qP "\t$name\t[0-9]+\tR49 formal" SUBMITTED_R49.tsv; then echo "$name already submitted"; return; fi
  until j=$(sbatch --parsable --qos=normal --partition=$ANY --exclude=node51,node53,node60 --nice=50 --mem=$mem --time=$tl \
      --export=ALL,R40_MEM=$mem,R40_TIME=$tl,R49_PART=$ANY -J $name sbatch/units_r49.sbatch $tree train_unit.py "$units" 2>/dev/null); do sleep 60; done
  printf "%s\t%s\t%s\tR49 formal (%s/train_unit.py, mem %s, time %s, partition %s, nice 50): %s\n" "$(date -u +%FT%TZ)" $name $j $tree $mem $tl $ANY "$units" >> SUBMITTED_R49.tsv; echo "$name $j"; }
sub r49_u_f r49f 64G 04:00:00
sub r49_u_o r49o 64G 06:00:00
sub r49_u_c r49c 64G 03:00:00
