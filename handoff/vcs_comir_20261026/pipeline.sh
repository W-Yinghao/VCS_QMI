#!/usr/bin/env bash
# R24 staged pipeline: smokes PASS -> feed 16 CAL -> pre-registered selection (select_lock.py) -> feed BF then EPFL finals -> analyze.py.
# Stops (no retry) on any failed attempt; touch STOP_PIPELINE in this directory to stop. Log: pipeline.log.
D=/home/infres/yinwang/CS_QMI/handoff/vcs_comir_20261026
F=/home/infres/yinwang/CS_QMI/handoff/review_claude_20261019/deps_watch
PY=/home/infres/yinwang/CS_QMI/env/bin/python
LOG=$D/pipeline.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
wait_stage() {   # wait until a stage is COMPLETE; exit on FAILED or STOP
  while :; do
    [ -e "$D/STOP_PIPELINE" ] && { log "STOP"; exit 0; }
    s=$(cd "$D" && "$PY" -B status.py "$1" 2>>"$LOG")
    case "$s" in
      COMPLETE*) log "$1 $s"; return 0 ;;
      FAILED*) log "$1 $s -- pipeline halted for review"; exit 1 ;;
    esac
    sleep 300
  done
}
cd "$D" || exit 2
log "pipeline start (pid $$)"
wait_stage smoke_bf
wait_stage smoke_epfl
bash "$F/app_lines_feed.sh" "$D/LINES_CAL.txt" r24_comir_cal 30 0
wait_stage cal
"$PY" -B select_lock.py >> "$LOG" 2>&1 || { log "select_lock failed"; exit 1; }
[ -f LOCK.json ] || { log "LOCK.json missing after selection"; exit 1; }
bash "$F/app_lines_feed.sh" "$D/LINES_FINAL_BF.txt" r24_comir_final_bf 30 0
bash "$F/app_lines_feed.sh" "$D/LINES_FINAL_EPFL.txt" r24_comir_final_epfl 30 0
wait_stage final_bf
wait_stage final_epfl
"$PY" -B analyze.py >> "$LOG" 2>&1 && log "analysis written (ANALYSIS.json)" || log "analyze.py failed"
log "pipeline done"
