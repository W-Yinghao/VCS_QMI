#!/usr/bin/env bash
# Login-node watcher for >23 h chains: holds each second link OUT of the queue (it would only occupy a normal-QOS submit slot) until its head job
# is RUNNING (or has left the queue), then submits it under the submit cap.  Lines of $1: "<head job id> <full sbatch command>"; the command should
# carry --dependency=singleton and the head's job name, so the link starts only after the head ends.  Log: slurm_logs/feed_normal.log.
# Run: nohup setsid bash slurm/link2_watch.sh <file> >/dev/null 2>&1 &   — stop by PID.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
CAP=${CAP:-30}; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/feed_normal.log
mapfile -t L < <(grep -vE '^\s*(#|$)' "$1"); declare -A done_
echo "$(date -u +%FT%TZ) link2_watch start pid $$ file=$1 entries=${#L[@]}" >> "$LOG"
while [ ${#done_[@]} -lt ${#L[@]} ]; do
  for i in "${!L[@]}"; do
    [ -n "${done_[$i]}" ] && continue
    head=${L[$i]%% *}; cmd=${L[$i]#* }; st=$(squeue -h -j "$head" -o %T 2>/dev/null)
    [ "$st" = "PENDING" ] && continue
    [ "$(squeue -h -u "$USER" -q normal | wc -l)" -ge "$CAP" ] && continue
    id=$(eval "$cmd" 2>>"$LOG" | tail -1)
    if [[ "$id" =~ ^[0-9]+$ ]]; then done_[$i]=1; echo "$(date -u +%FT%TZ) link2 submitted $id after head $head (${st:-gone}) : $(echo "$cmd" | grep -oE -- '--job-name=[^ ]+')" >> "$LOG"
    else echo "$(date -u +%FT%TZ) link2 retry (head $head): $cmd" >> "$LOG"; fi
  done
  sleep 120
done
echo "$(date -u +%FT%TZ) link2_watch done" >> "$LOG"
