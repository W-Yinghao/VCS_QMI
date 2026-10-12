#!/usr/bin/env bash
# Keeps filler jobs behind main-route units: every 60 s, any PENDING job of mine whose name matches $PAT (default ^r6reg3_) and still has
# Nice 0 gets Nice $NICE (default 300; EligibleTime unchanged; ordering only).  Stops when no matching job has been seen for 30 min.  Stop by PID.
PAT=${PAT:-^r6reg3_}; NICE=${NICE:-300}; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/nice_filler.log; echo "$(date -u +%FT%TZ) start pid $$ pattern $PAT nice $NICE" >> $LOG; idle=0
while [ $idle -lt 30 ]; do
  any=$(squeue -h -u yinwang -q normal -o "%j" | grep -cE "$PAT"); [ "$any" -eq 0 ] && idle=$((idle + 1)) || idle=0
  for j in $(squeue -h -u yinwang -q normal -t PD -o "%i %j %y" | awk -v p="$PAT" '$2 ~ p && $3 == 0 {print $1}'); do scontrol update jobid=$j Nice=$NICE && echo "$(date -u +%FT%TZ) nice $NICE -> $j" >> $LOG; done
  sleep 60
done; echo "$(date -u +%FT%TZ) done" >> $LOG
