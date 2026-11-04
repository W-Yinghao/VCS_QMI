#!/usr/bin/env bash
# usage: submit_units.sh Flickr00 Flickr60 ...   (submits complete six-unit matched blocks, deduplicated)
set -euo pipefail
H=/home/infres/yinwang/CS_QMI/handoff/nr1_20261104; O=/home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104
cd $H
[ -f $O/cache/fit/manifest.json ] && [ -f $O/cache/dev/manifest.json ] || { echo "caches missing"; exit 2; }
for c in "$@"; do
  [ -f $O/calibration/$c.json ] || { echo "calibration $c missing"; exit 2; }
  for a in G GL; do for l in J BCE NCE; do
    U=NR1_${c}_${a}_${l}_s1001
    if [ -f $O/runs/$U/UNIT_COMPLETE.json ]; then echo "skip complete $U"; continue; fi
    if squeue -u yinwang -h -o %j | grep -qx "nr1_$U"; then echo "skip queued $U"; continue; fi
    mem=16G; [ $a = GL ] && mem=48G
    j=$(sbatch --parsable --job-name=nr1_$U --mem=$mem sbatch/unit.sbatch $U)
    printf "%s\tnr1_%s\t%s\tformal unit (train 8000 + DEV eval 7 steps)\tRTX6000PRO,L40S,H100/normal nice50 mem%s\n" $j $U "$(date -u +%FT%TZ)" $mem >> SUBMITTED_NR1.tsv
    echo "$U $j"
  done; done
done
