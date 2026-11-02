#!/bin/bash
# R37S critic formal: 18 units in 3 normal nice-0 GPU jobs (N shared, N directional, T shared; constant + cosine), after PROTOCOL_FROZEN_CRITIC.json. DRY=1 prints only.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r37s_cc152k_r30h_20261102; cd $W
test -f PROTOCOL_FROZEN_CRITIC.json
if [ -z "${DRY:-}" ] && squeue -u yinwang -h -o %j | grep -q '^r37s_u_'; then echo "R37S critic jobs already queued"; exit 1; fi
P=r37s_cc_r30h_s1001
sub() { local name=$1 units=$2 j; [ -n "${DRY:-}" ] && { echo "$name $units"; return; }; j=$(sbatch --parsable --qos=normal --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 --nice=0 -J $name sbatch/critic_units.sbatch r37s_critic "$units"); printf "%s\t%s\t%s\tformal critic (r37s_critic): %s\n" "$(date -u +%FT%TZ)" $name $j "$units" >> SUBMITTED.tsv; echo "$name $j"; }
L() { local r=$1 a=$2 out=""; shift 2; for s in "$@"; do for o in J BCE NCE; do out="$out,${P}_${r}_${a}_${o}_${s}"; done; done; echo ${out#,}; }
sub r37s_u_N_shared "$(L N shared constant cosine)"
sub r37s_u_N_dir "$(L N directional constant cosine)"
sub r37s_u_T_shared "$(L T shared constant cosine)"
