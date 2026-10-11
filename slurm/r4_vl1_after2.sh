#!/usr/bin/env bash
# Replaces r4_vl1_after.sh: once no R4-VL1 run AND no R4-VL0 CAL-scoring job is pending, restore Nice=0 on pending VL3 add. 7 runs and feed the 6 owed
# add. 7 runs with CAP=27 (owner headroom).  Login node; stop by PID.
cd /home/infres/yinwang/CS_QMI/ssl_pilot; L=/home/infres/yinwang/CS_QMI/slurm_logs/feed_normal.log
while squeue -h -u "$USER" -t PD -o "%j" | grep -qE "^(r4vl1_|r4_vl0cal_)"; do sleep 120; done
n=0; for j in $(squeue -h -u "$USER" -q normal -t PD -o "%i %j %y" | awk '$2 ~ /^vl3e20/ && $3 != 0 {print $1}'); do scontrol update jobid=$j Nice=0 && n=$((n+1)); done
echo "$(date -u +%FT%TZ) R4-VL1 and CAL scoring all started: Nice restored on $n add. 7 runs; feeding the 6 owed add. 7 runs (CAP 27)" >> $L
CAP=27 bash slurm/feed_normal.sh slurm/vl3_add7_owed.txt
