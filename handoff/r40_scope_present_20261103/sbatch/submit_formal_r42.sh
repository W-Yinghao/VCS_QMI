#!/bin/bash
# R42 formal submission after CPU_TEST_R42 ALL_PASS: freeze, then one job per (parent, cell) running J/BCE/NCE; normal nice 50, RTX6000PRO/L40S/H100
# excl. node51/53/60; frozen 64G/2h, joint 64G/8h; self-resume <= 3 (units_r42.sbatch). Oracle cells: the frozen and joint oracle jobs share
# OUT/oracle_runs_r42/ORACLE_DIAGNOSTICS.json, so only ONE oracle job is submitted here (ORACLE=frozen|joint, default joint); the other after it ends.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103; cd $W; PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
PART=${1:-1}; F=$([ $PART = 1 ] && echo PROTOCOL_FROZEN_R42.json || echo PROTOCOL_FROZEN_R42B.json)
[ -n "${DRY:-}" ] || [ -f $F ] || $PY -I -B tools/freeze_r42.py $PART
sub() { local name=$1 runner=$2 mem=$3 tl=$4 units=$5 j
  if [ -z "${DRY:-}" ] && squeue -u yinwang -h -o %j | grep -qx "$name"; then echo "$name already queued"; return; fi
  [ -n "${DRY:-}" ] && { echo "$name $runner $mem $tl $units"; return; }
  j=$(sbatch --parsable --qos=normal --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 --nice=50 --mem=$mem --time=$tl \
      --export=ALL,R40_MEM=$mem,R40_TIME=$tl -J $name sbatch/units_r42.sbatch r42f $runner "$units")
  printf "%s\t%s\t%s\tR42 formal (r42f/%s, mem %s, time %s, nice 50): %s\n" "$(date -u +%FT%TZ)" $name $j $runner $mem $tl "$units" >> SUBMITTED_R42.tsv; echo "$name $j"; }
three() { echo "${1}_J,${1}_BCE,${1}_NCE"; }
PS=$([ $PART = 1 ] && echo "F20_IN2R_H1001 F20_IN2R_H1002 F20_IN2R_H1003 F40_IN2R_H1002 F80_IN2R_H1002 F80_IN2R_H1003" || echo F40_IN2R_H1003)
if [ "${ORACLE_ONLY:-}" = "" ]; then
  for P in $PS; do
    sub r42_u_${P}_joint train_unit.py 64G 08:00:00 "$(three R42_N_${P}_N_joint)"
    sub r42_u_${P}_frozen train_unit.py 64G 02:00:00 "$(three R42_N_${P}_N_frozen)"; done
fi
[ $PART = 1 ] || exit 0
O=${ORACLE:-joint}; TL=$([ $O = joint ] && echo 08:00:00 || echo 02:00:00)
sub r42_u_O_F80_IN2R_H1001_oracle_$O train_oracle.py 64G $TL "$(three R42_O_F80_IN2R_H1001_ORACLE_$O)"
