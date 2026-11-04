#!/usr/bin/env bash
# Submit the 36 NR2 formal units (dedup: UNIT_COMPLETE or same job name already queued). Requires PROTOCOL_FROZEN_NR2.json.
set -euo pipefail
N=/home/infres/yinwang/CS_QMI/handoff/nr2_20261104; O=/home/infres/yinwang/CS_QMI/outputs_nr2/NR2_20261104; cd $N
[ -f PROTOCOL_FROZEN_NR2.json ] || { echo "not frozen"; exit 2; }
for f in locked_configs/NR2_*.json; do
  U=$(basename $f .json)
  /home/infres/yinwang/CS_QMI/env_nr1/bin/python -I -B code/site/check_frozen.py $U >/dev/null
  if [ -f $O/runs/$U/UNIT_COMPLETE.json ]; then echo "skip complete $U"; continue; fi
  if squeue -u yinwang -h -o %j | grep -qx "nr2_$U"; then echo "skip queued $U"; continue; fi
  j=$(sbatch --parsable --job-name=nr2_$U sbatch/unit.sbatch $U)
  printf "%s\tnr2_%s\t%s\tformal unit (train 8000 + FIT readout + DEV eval 7 steps + eval-only FIT strata)\tRTX6000PRO,L40S,H100/normal nice50 mem24G 2h\n" $j $U "$(date -u +%FT%TZ)" >> SUBMITTED_NR2.tsv
  echo "$U $j"
done
