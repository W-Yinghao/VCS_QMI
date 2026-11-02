#!/bin/bash
# R38 user amendment: 54 units in 8 normal nice-0 GPU jobs (after AMENDMENT_CPU_CHECK + amended freeze). DRY=1 prints only.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102; cd $W
if [ -z "${DRY:-}" ] && squeue -u yinwang -h -o %j | grep -q '^r38_a_'; then echo "R38 amendment jobs already queued"; exit 1; fi
sub() { local name=$1 tree=$2 units=$3 j; [ -n "${DRY:-}" ] && { echo "$name $tree $units"; return; }; j=$(sbatch --parsable --qos=normal --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 --nice=0 -J $name sbatch/units.sbatch $tree "$units"); printf "%s\t%s\t%s\tformal amendment (%s): %s\n" "$(date -u +%FT%TZ)" $name $j $tree "$units" >> SUBMITTED.tsv; echo "$name $j"; }
for pk in f60_R30_s1002 f60_R30_s1003 cc_R31_s1002 cc_R31_s1003; do
  case $pk in f60*) tree=r38f; tag=f${pk: -1};; *) tree=r38c; tag=c${pk: -1};; esac
  N=""; for c in shared_%s_cosine directional_%s_constant directional_%s_cosine; do for o in J BCE NCE; do N="$N,r38_B_${pk}_N_$(printf $c $o)"; done; done
  T=""; for o in J BCE NCE; do T="$T,r38_C_${pk}_T_shared_${o}_cosine"; done
  case $pk in cc*) for o in J BCE NCE; do T="$T,r38_C_${pk}_T_shared_${o}_constant"; done;; esac
  sub r38_a_${tag}_N $tree "${N#,}"; sub r38_a_${tag}_T $tree "${T#,}"
done
