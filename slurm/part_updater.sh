#!/usr/bin/env bash
# Keeps matching PENDING jobs of mine off a partition without restarting a running driver: every $INTERVAL s, each pending normal-QOS job
# whose name matches $PAT and whose partition list differs from $PART gets Partition=$PART (EligibleTime unchanged; placement only).
# Stops after 2 h without a matching job.  Stop by PID.  Log: slurm_logs/part_updater.log
PAT=${PAT:?set PAT}; PART=${PART:?set PART}; INTERVAL=${INTERVAL:-5}; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/part_updater.log; idle=0
echo "$(date -u +%FT%TZ) start pid $$ pattern $PAT partition $PART interval $INTERVAL s" >> $LOG
while [ $idle -lt $(( 7200 / INTERVAL )) ]; do
  any=$(squeue -h -u yinwang -q normal -o "%j" | grep -cE "$PAT"); [ "$any" -eq 0 ] && idle=$((idle + 1)) || idle=0
  for j in $(squeue -h -u yinwang -q normal -t PD -o "%i %j %P" | awk -v p="$PAT" -v q="$PART" '$2 ~ p && $3 != q {print $1}'); do
    scontrol update jobid="$j" Partition="$PART" && echo "$(date -u +%FT%TZ) $j -> $PART" >> $LOG; done
  sleep "$INTERVAL"
done; echo "$(date -u +%FT%TZ) done" >> $LOG
