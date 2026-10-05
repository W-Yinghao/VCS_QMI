#!/usr/bin/env bash
# Login-node feeder for the normal-QOS submit cap (30 jobs per user; runfill jobs of the owner do not count).  Each non-comment line of $1 is a full
# shell command ending in `sbatch --parsable …`; submitted in order whenever my normal-QOS job count is below CAP.  Log: slurm_logs/feed_normal.log.
# Run: nohup setsid bash slurm/feed_normal.sh <lines_file> >/dev/null 2>&1 &   — stop by PID.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
LINES=$1; CAP=${CAP:-30}; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/feed_normal.log
echo "$(date -u +%FT%TZ) start pid $$ lines=$LINES cap=$CAP" >> "$LOG"
while IFS= read -r line || [ -n "$line" ]; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  while :; do
    while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -ge "$CAP" ]; do sleep 120; done
    id=$(eval "$line" 2>>"$LOG" | tail -1)
    [[ "$id" =~ ^[0-9]+$ ]] && break
    echo "$(date -u +%FT%TZ) retry: $line" >> "$LOG"; sleep 120
  done
  echo "$(date -u +%FT%TZ) submitted $id : $(echo "$line" | grep -oE -- '--job-name=[^ ]+|slurm/[a-z0-9_]+\.sbatch' | tr '\n' ' ')" >> "$LOG"
done < "$LINES"
echo "$(date -u +%FT%TZ) done" >> "$LOG"
