#!/usr/bin/env bash
# Gated feeder for backlog filler: like slurm/feed_normal.sh, but each line is submitted only when (a) my normal-QOS job count is below CAP and
# (b) my pending GPU backlog is small (fewer than GATE pending GPU jobs), so filler never holds submit slots that main-route units need.
# Run: CAP=27 GATE=3 nohup setsid bash slurm/feed_gated.sh <lines_file> >/dev/null 2>&1 &   — stop by PID.  Log: slurm_logs/feed_gated.log.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
LINES=$1; CAP=${CAP:-27}; GATE=${GATE:-3}; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/feed_gated.log
echo "$(date -u +%FT%TZ) start pid $$ lines=$LINES cap=$CAP gate=$GATE" >> "$LOG"
while IFS= read -r line || [ -n "$line" ]; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  while :; do
    n=$(squeue -h -u "$USER" -q normal | wc -l); pd=$(squeue -h -u "$USER" -q normal -t PD -o "%b" | grep -cE "gpu:[1-9]")
    if [ "$n" -lt "$CAP" ] && [ "$pd" -lt "$GATE" ]; then
      id=$(eval "$line" 2>>"$LOG" | tail -1); [[ "$id" =~ ^[0-9]+$ ]] && break
      echo "$(date -u +%FT%TZ) retry: $line" >> "$LOG"
    fi
    sleep 120
  done
  echo "$(date -u +%FT%TZ) submitted $id : $(echo "$line" | grep -oE -- '--job-name=[^ ]+' | tr '\n' ' ')" >> "$LOG"
done < "$LINES"
echo "$(date -u +%FT%TZ) done" >> "$LOG"
