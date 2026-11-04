#!/bin/bash
cd /home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103
U=R47_G_CC31_H1001_N_joint_BCE,R47_S_CC31_H1001_N_joint_J
while true; do
  j=$(sbatch --parsable --qos=normal --partition=RTX6000PRO,L40S,H100 --exclude=node51,node53,node60 --nice=0 --mem=150G -J r47_smoke_c sbatch/r47_smoke.sbatch r47c "$U" 2>/dev/null)
  if [ -n "$j" ]; then printf "%s\tr47_smoke_c\t%s\tR47 GPU smoke (gate mode, 20 updates/unit, r47c, mem 150G): %s\n" "$(date -u +%FT%TZ)" $j "$U" >> SUBMITTED_R47.tsv; echo "submitted $j"; exit 0; fi
  sleep 60
done
