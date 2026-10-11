#!/usr/bin/env bash
# After every R4-VL1 run is submitted (feeder PID $1 gone) and started (no r4vl1_* pending): restore Nice=0 on pending VL3 add. 7 runs, then feed
# the 6 owed add. 7 runs with CAP=27 (owner headroom).  Login node; stop by PID.
cd /home/infres/yinwang/CS_QMI/ssl_pilot; L=/home/infres/yinwang/CS_QMI/slurm_logs/feed_normal.log
while kill -0 "$1" 2>/dev/null; do sleep 120; done
while squeue -h -u "$USER" -t PD -o "%j" | grep -q "^r4vl1_"; do sleep 120; done
n=0; for j in $(squeue -h -u "$USER" -q normal -t PD -o "%i %j %y" | awk '$2 ~ /^vl3e20/ && $3 != 0 {print $1}'); do scontrol update jobid=$j Nice=0 && n=$((n+1)); done
echo "$(date -u +%FT%TZ) R4-VL1 all started: Nice restored on $n add. 7 runs; feeding the 6 owed add. 7 runs (CAP 27)" >> $L
CAP=27 bash slurm/feed_normal.sh slurm/vl3_add7_owed.txt
