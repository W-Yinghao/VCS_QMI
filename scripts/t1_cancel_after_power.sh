#!/usr/bin/env bash
# P73 addendum 1: cancel a monolithic T1 job once its last planted cell (n=2000) is logged.
L=/home/infres/yinwang/CS_QMI/slurm_logs
declare -A last=([1013022]="colour_s0.2] n=2000" [1013024]="colour_s0.2] n=2000" [1013025]="blur_s0.5] n=2000")
while ((${#last[@]})); do
  for j in "${!last[@]}"; do
    f=$(ls $L/cond_test_t1_*_$j.out)
    if grep -qF "[${last[$j]}" "$f"; then scancel $j && echo "$(date -u +%FT%TZ) cancelled $j after power cells"; unset "last[$j]"
    elif ! squeue -h -j $j | grep -q .; then echo "$(date -u +%FT%TZ) $j left the queue before its last power cell"; unset "last[$j]"; fi
  done; sleep 120
done
