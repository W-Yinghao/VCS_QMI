#!/bin/bash
# Wait for the 6 KD lambda candidates (S1, CAL seed 27182), apply the fixed selection rule (select_kd.py), then feed the
# 6 final lines with app_lines_feed.sh (TAG r22_kd). Polls every 5 min; exits on selection or when a manual decision is
# needed (select_kd.py exit 3). STOP_R22_KD_AFTER_CAL in deps_watch makes it exit. No --dependency jobs.
# Usage: kd_after_cal.sh S1_CODE_DIR S2_CODE_DIR
set -uo pipefail
source /home/infres/yinwang/CS_QMI/ssl_pilot_app_r22/apps/r22/server/paths.env
S1=$1; S2=$2; D=/home/infres/yinwang/CS_QMI/handoff/review_claude_20261019/deps_watch; LOG=$D/watch.log
SEL=$R22_OUT/kd/kd_lambda_selection.json; LINES=$R22_WT/apps/r22/server/lines/KD_S2_LINES.txt
log() { echo "$(date -u +%FT%TZ) r22_kd_after_cal: $*" >> "$LOG"; }
log "start (pid $$): S1 $S1, S2 $S2"
while :; do
  [ -e "$D/STOP_R22_KD_AFTER_CAL" ] && { log "STOP"; exit 0; }
  msg=$("$KD_PY" -I "$R22_WT/apps/r22/select_kd.py" "$S1" "$S2" "$SEL" "$LINES" 2>&1); rc=$?
  case $rc in
    0) log "selected: $msg"; cd "$D" && exec bash ./app_lines_feed.sh "$LINES" r22_kd 28 1587 ;;
    2) : ;;
    *) log "needs a manual decision (rc $rc): $msg"; exit 3 ;;
  esac
  sleep 300
done
