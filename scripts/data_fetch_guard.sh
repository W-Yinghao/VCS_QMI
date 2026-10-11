#!/usr/bin/env bash
# Free-space guard for the data_fetch.py runners (the runners only check the floor when an item starts).
# Every 60 s: if free space on the shared filesystem drops below FLOOR_GB, stop the given runner PIDs and all their
# descendants (wget, git) by PID, log it, and exit. Downloads resume with `data_fetch.py run` (wget -c / HF resume).
# Usage: nohup setsid scripts/data_fetch_guard.sh ROOT_PID [ROOT_PID ...] &
FS=/projects/EEG-foundation-model; FLOOR_GB=${FLOOR_GB:-1205}; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/data_fetch.log
desc() { local p; for p in $(pgrep -P "$1"); do echo "$p"; desc "$p"; done; }
echo "$(date -u +%FT%TZ) GUARD start roots=$* floor=${FLOOR_GB} GB pid=$$" >> "$LOG"
while :; do
  alive=(); for p in "$@"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
  [ ${#alive[@]} -eq 0 ] && { echo "$(date -u +%FT%TZ) GUARD exit: runners finished" >> "$LOG"; exit 0; }
  free=$(( $(df -B1 --output=avail "$FS" | tail -1) / 1000000000 ))
  if [ "$free" -lt "$FLOOR_GB" ]; then
    all=(); for p in "${alive[@]}"; do all+=("$p" $(desc "$p")); done
    kill -TERM "${all[@]}" 2>/dev/null
    echo "$(date -u +%FT%TZ) GUARD free ${free} GB < ${FLOOR_GB} GB: stopped PIDs ${all[*]}" >> "$LOG"; exit 1
  fi
  sleep 60
done
