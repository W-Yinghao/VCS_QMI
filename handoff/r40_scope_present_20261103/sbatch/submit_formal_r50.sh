#!/bin/bash
# R50 formal submission after CPU_TEST_R50 ALL_PASS and the three GPU smokes ALL_PASS: freeze (args: smoke job ids f c oracle), check_frozen, then one job per
# cell running J/BCE/NCE (8 jobs, 24 units); normal nice 50; RTX6000PRO,L40S,H100 excl node51/53/60; self-resume <= 3. Oracle cells through r50f/train_oracle.py
# (per-job diagnostics file, no shared write -> parallel). Flickr frozen 64G/3h, Flickr joint 64G/8h, CC frozen 64G/3h, CC joint 150G/10h (R47 / R48 precedents).
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103; cd $W; PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
[ -f PROTOCOL_FROZEN_R50.json ] || $PY -I -B tools/freeze_r50.py $1 $2 $3
for t in r50f r50c; do (cd /tmp && $PY -I -B -c "
import sys,json; W='$W'; sys.path.insert(0,W+'/$t'); import train_unit as TU
u=[x for x in json.load(open(W+'/configs/UNITS_R50.json')) if json.load(open(f'{W}/configs/{x}.json'))['code_tree']=='$t']; TU.check_frozen(u); print('$t check_frozen OK', len(u))"); done
ANY=RTX6000PRO,L40S,H100
sub() { local name=$1 tree=$2 runner=$3 mem=$4 tl=$5 units=$6 j
  if squeue -u yinwang -h -o %j | grep -qx "$name"; then echo "$name already queued"; return; fi
  if grep -qP "\t$name\t[0-9]+\tR50 formal" SUBMITTED_R50.tsv; then echo "$name already submitted"; return; fi
  until j=$(sbatch --parsable --qos=normal --partition=$ANY --exclude=node51,node53,node60 --nice=50 --mem=$mem --time=$tl \
      --export=ALL,R40_MEM=$mem,R40_TIME=$tl,R50_PART=$ANY -J $name sbatch/units_r50.sbatch $tree $runner "$units" 2>/dev/null); do sleep 60; done
  printf "%s\t%s\t%s\tR50 formal (%s/%s, mem %s, time %s, partition %s, nice 50): %s\n" "$(date -u +%FT%TZ)" $name $j $tree $runner $mem $tl $ANY "$units" >> SUBMITTED_R50.tsv; echo "$name $j"; }
three() { echo "${1}_J,${1}_BCE,${1}_NCE"; }
sub r50_u_A_F60_frozen r50f train_unit.py 64G 03:00:00 "$(three R50_A_F60_R30H1001_PRESENT_ONLY_frozen)"
sub r50_u_A_F60_joint r50f train_unit.py 64G 08:00:00 "$(three R50_A_F60_R30H1001_PRESENT_ONLY_both_joint)"
sub r50_u_A_CC_frozen r50c train_unit.py 64G 03:00:00 "$(three R50_A_CC_R31H1001_PRESENT_ONLY_frozen)"
sub r50_u_A_CC_joint r50c train_unit.py 150G 10:00:00 "$(three R50_A_CC_R31H1001_PRESENT_ONLY_both_joint)"
sub r50_o_B_F60_frozen r50f train_oracle.py 64G 03:00:00 "$(three R50_B_F60_R30H1001_ORACLE_MISSING_frozen)"
sub r50_o_B_F60_joint r50f train_oracle.py 64G 08:00:00 "$(three R50_B_F60_R30H1001_ORACLE_MISSING_both_joint)"
sub r50_o_B_F80_frozen r50f train_oracle.py 64G 03:00:00 "$(three R50_B_F80_IN2RH1001_ORACLE_MISSING_frozen)"
sub r50_o_B_F80_joint r50f train_oracle.py 64G 08:00:00 "$(three R50_B_F80_IN2RH1001_ORACLE_MISSING_both_joint)"
