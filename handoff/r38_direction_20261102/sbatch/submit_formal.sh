#!/bin/bash
# R38 formal submission (once, after CPU_TEST + PARENT_ARTIFACT_CHECK ALL_PASS): freeze, then 6 normal nice-0 GPU jobs = 30 units.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r38_direction_20261102; cd $W
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
[ -n "${DRY:-}" ] || [ -f PROTOCOL_FROZEN.json ] || $PY -I -B tools/freeze.py
if [ -z "${DRY:-}" ] && squeue -u yinwang -h -o %j | grep -q '^r38_u_'; then echo "R38 formal jobs already queued"; exit 1; fi
sub() { local name=$1 tree=$2 units=$3 j; [ -n "${DRY:-}" ] && { echo "$name $tree $units"; return; }; j=$(sbatch --parsable --qos=normal --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 --nice=0 -J $name sbatch/units.sbatch $tree "$units"); printf "%s\t%s\t%s\tformal (%s): %s\n" "$(date -u +%FT%TZ)" $name $j $tree "$units" >> SUBMITTED.tsv; echo "$name $j"; }
B=r38_B_f60_R30_s1001_N; C=r38_C_f60_R30_s1001_T
sub r38_u_f60_N r38f "${B}_shared_J_cosine,${B}_shared_BCE_cosine,${B}_shared_NCE_cosine,${B}_directional_J_constant,${B}_directional_BCE_constant,${B}_directional_NCE_constant,${B}_directional_J_cosine,${B}_directional_BCE_cosine,${B}_directional_NCE_cosine"
sub r38_u_f60_T r38f "${C}_shared_J_cosine,${C}_shared_BCE_cosine,${C}_shared_NCE_cosine"
B=r38_B_cc_R31_s1001_N; C=r38_C_cc_R31_s1001_T
sub r38_u_cc1_N r38c "${B}_shared_J_cosine,${B}_shared_BCE_cosine,${B}_shared_NCE_cosine,${B}_directional_J_constant,${B}_directional_BCE_constant,${B}_directional_NCE_constant,${B}_directional_J_cosine,${B}_directional_BCE_cosine,${B}_directional_NCE_cosine"
sub r38_u_cc1_T r38c "${C}_shared_J_cosine,${C}_shared_BCE_cosine,${C}_shared_NCE_cosine"
for s in 1002 1003; do A=r38_A_cc_R31_s${s}_N_shared; sub r38_u_cc${s: -1}_N r38c "${A}_J_constant,${A}_BCE_constant,${A}_NCE_constant"; done
