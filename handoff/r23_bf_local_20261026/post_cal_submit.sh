#!/usr/bin/env bash
# After all 21 CAL runs are COMPLETE, run the frozen pre-registered `run.py select` (no DEV read) and feed the generated
# LINES_PILOT/LINES_EVAL at nice 0 through the existing line feeder. Closes the idle gap between CAL completion and the
# 2-hourly patrol. Touch STOP_POST_CAL in this directory to stop before LOCK.json exists.
D=/home/infres/yinwang/CS_QMI/handoff/r23_bf_local_20261026
F=/home/infres/yinwang/CS_QMI/handoff/review_claude_20261019/deps_watch
PY=/home/infres/yinwang/CS_QMI/env/bin/python
LOG=$D/post_cal.log
cd "$D" || exit 2
echo "$(date -u +%FT%TZ) post_cal start (pid $$)" >> "$LOG"
while [ ! -f LOCK.json ]; do
  [ -e STOP_POST_CAL ] && { echo "$(date -u +%FT%TZ) STOP" >> "$LOG"; exit 0; }
  "$PY" -I run.py select >> "$LOG" 2>&1
  [ -f LOCK.json ] || sleep 300
done
echo "$(date -u +%FT%TZ) LOCK.json written; feeding pilot and eval lines" >> "$LOG"
for k in PILOT EVAL; do
  [ -f "LINES_$k.txt" ] || { echo "$(date -u +%FT%TZ) missing LINES_$k.txt" >> "$LOG"; continue; }
  sed 's/^sbatch /sbatch --nice=0 /' "LINES_$k.txt" > "LINES_${k}_NICE0.txt"
  bash "$F/app_lines_feed.sh" "$D/LINES_${k}_NICE0.txt" "r23_$(echo $k | tr 'A-Z' 'a-z')" 30 0
done
echo "$(date -u +%FT%TZ) post_cal done" >> "$LOG"
