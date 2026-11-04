#!/bin/bash
# submit corrected-readout groups (one normal-QOS job each, on the training GPU type of the group; nice 50); skips groups with a result or a queued job.
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103; cd $W
for g in ${@:-$(ls corrected/groups/*.json)}; do n=$(basename $g .json); [ -f corrected/results/$n.json ] && continue
  squeue -u yinwang -h -o %j | grep -qx "cr_$n" && continue
  case $n in *__RTX6000PRO) P=RTX6000PRO;; *__L40S) P=L40S;; *__NVL) P=H100;; *) P=RTX6000PRO,L40S,H100;; esac
  case $n in r46o__*|r44c__*|r47cca__*) M=160G;; *) M=64G;; esac
  until [ $(squeue -u yinwang -h | wc -l) -lt 30 ]; do sleep 60; done
  j=$(sbatch --parsable --qos=normal --partition=$P --exclude=node51,node53,node60 --nice=50 --mem=$M -J cr_$n sbatch/corrected.sbatch $W/$g) || continue
  printf "%s\t%s\t%s\t%s\n" "$(date -u +%FT%TZ)" cr_$n $j "corrected layout-free DEV readouts, group $n (partition $P, mem $M, nice 50)" >> SUBMITTED_CORRECTED.tsv; echo "cr_$n $j"
done
