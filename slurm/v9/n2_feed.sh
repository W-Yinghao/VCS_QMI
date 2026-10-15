#!/usr/bin/env bash
# V9-N2 login-node feeder for the normal-QOS cap (owner: `squeue -u $USER -h -o %q | grep -c normal` <= 27; runfill jobs do not count).
# Each non-comment line of $1 is a full shell command ending in `sbatch --parsable ...`; lines are submitted in order whenever the count is
# below CAP (default 27, i.e. the new job keeps the total <= 27).  Log: slurm_logs/v9_n2_feed.log; submitted ids: slurm_logs/v9_n2_feed_ids.txt.
# Run: nohup setsid bash slurm/v9/n2_feed.sh <lines_file> >/dev/null 2>&1 &   (stop by PID; the PID is logged)
cd /home/infres/yinwang/CS_QMI/ssl_pilot_v9 || exit 1
LINES=$1; CAP=${CAP:-27}; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/v9_n2_feed.log; IDS=/home/infres/yinwang/CS_QMI/slurm_logs/v9_n2_feed_ids.txt
echo "$(date -u +%FT%TZ) start pid $$ lines=$LINES cap=$CAP" >> "$LOG"
while IFS= read -r line || [ -n "$line" ]; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  while :; do
    while [ "$(squeue -u "$USER" -h -o %q | grep -c normal)" -ge "$CAP" ]; do sleep 120; done
    id=$(eval "$line" 2>>"$LOG" | tail -1)
    [[ "$id" =~ ^[0-9]+$ ]] && break
    echo "$(date -u +%FT%TZ) retry: $line" >> "$LOG"; sleep 300
  done
  echo "$(date -u +%FT%TZ) submitted $id : $(echo "$line" | grep -oE -- '--job-name=[^ ]+|group [^ ]+' | tr '\n' ' ')" >> "$LOG"; echo "$id $(echo "$line" | grep -oE -- 'group [^ ]+')" >> "$IDS"
  sleep 20
done < "$LINES"
echo "$(date -u +%FT%TZ) done pid $$ lines=$LINES" >> "$LOG"
