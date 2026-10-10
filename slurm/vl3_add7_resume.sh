#!/usr/bin/env bash
# After every VL1-22 Table C job has left PENDING: restore Nice=0 on my pending add. 7 jobs and resume the add. 7 feed (rest file).  Login node; stop by PID.
cd /home/infres/yinwang/CS_QMI/ssl_pilot; L=/home/infres/yinwang/CS_QMI/slurm_logs
while squeue -h -u "$USER" -t PD -o "%j" | grep -q "^vl1_22_"; do sleep 120; done
for j in $(squeue -h -u "$USER" -q normal -t PD -o "%i %j %y" | awk '$2 ~ /^vl3e20/ && $3 != 0 {print $1}'); do scontrol update jobid=$j Nice=0; done
echo "$(date -u +%FT%TZ) add. 7: Table C all started - Nice restored, feed resumed" >> $L/feed_normal.log
bash slurm/feed_normal.sh slurm/vl3_add7_feed_rest.txt
