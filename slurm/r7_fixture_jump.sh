#!/usr/bin/env bash
# Temporary reordering of my own pending jobs (cluster policy: Nice=200 then Nice=0; EligibleTime unchanged): lets the two short r7 fixture
# GPU jobs start before the pending r6reg jobs; restores Nice=0 on every pending r6reg job once both are RUNNING or gone.  Stop by PID.
PAT=${PAT:-^r6reg_}; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/r7_fixture_jump.log; echo "$(date -u +%FT%TZ) start pid $$ watching $* (pattern $PAT)" >> $LOG
while true; do
  waiting=0; for j in "$@"; do s=$(squeue -h -j $j -o %T 2>/dev/null); [ "$s" = "PENDING" ] && waiting=1; done
  # keep newly submitted r6reg jobs behind as well
  for j in $(squeue -h -u yinwang -q normal -t PD -o "%i %j %y" | awk -v p="$PAT" '$2 ~ p && $3 == 0 {print $1}'); do [ $waiting = 1 ] && scontrol update jobid=$j Nice=200; done
  [ $waiting = 0 ] && break; sleep 30
done
for j in $(squeue -h -u yinwang -q normal -t PD -o "%i %j %y" | awk -v p="$PAT" '$2 ~ p && $3 != 0 && $3 != 300 {print $1}'); do scontrol update jobid=$j Nice=0; done
echo "$(date -u +%FT%TZ) r7 jobs started or gone; Nice restored to 0 on pending jobs matching $PAT (Nice 300 backlog untouched)" >> $LOG
