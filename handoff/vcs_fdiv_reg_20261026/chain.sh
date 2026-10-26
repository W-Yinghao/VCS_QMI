#!/usr/bin/env bash
# Stage chain (no --dependency): gate PASS -> 4 CAL jobs -> select (LOCK.json, CPU, no DEV read) -> 5 DEV jobs -> analyze.
# Each stage starts only after the previous stage's COMPLETE records exist; any FAILED job stops the chain (no retry).
# Touch STOP_CHAIN in this directory to stop.
D=/home/infres/yinwang/CS_QMI/handoff/vcs_fdiv_reg_20261026; PY=/home/infres/yinwang/CS_QMI/env/bin/python; LOG=$D/chain.log
SB="sbatch --parsable"; cd /home/infres/yinwang/CS_QMI/slurm_logs || exit 2
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
alive() { [ -n "$(squeue -j "$1" -h -o %T 2>/dev/null)" ]; }
waitjobs() { for j in "$@"; do while alive "$j"; do [ -e "$D/STOP_CHAIN" ] && { log STOP; exit 0; }; sleep 120; done; done; }
submit() { local name=$1; shift; local j; for a in 1 2 3 4 5; do j=$($SB --job-name="$name" "$@" 2>>"$LOG") && { echo "$j"; return; }; sleep 120; done; log "submit failed $name"; exit 3; }
log "chain start pid $$"
waitjobs 1066348
grep -q '"status": "PASS' "$D/GATE.json" 2>/dev/null || { log "gate not PASS; stop"; exit 4; }
CAL=()
for spec in "bf mi,nmi,vcs,js" "bf chi2,cs_qmi,hellinger" "epfl mi,nmi,vcs,js" "epfl chi2,cs_qmi,hellinger"; do
  set -- $spec; j=$(submit "r24_fdiv_cal_$1_${2//,/-}" --time=03:00:00 "$D/gpu.sbatch" cal --ds "$1" --divs "$2"); CAL+=("$j"); log "CAL $spec -> $j"
done
waitjobs "${CAL[@]}"
n=$(ls "$D"/runs/cal_*/COMPLETE.json 2>/dev/null | wc -l); [ "$n" -eq 4 ] || { log "CAL incomplete ($n/4); stop"; exit 5; }
"$PY" -I -B "$D/run.py" select >> "$LOG" 2>&1 || { log "select failed"; exit 6; }
DEV=()
for spec in "bf mi,vcs,js,nmi" "bf mi,chi2,cs_qmi,hellinger" "epfl mi,vcs,js" "epfl mi,nmi,chi2" "epfl mi,cs_qmi,hellinger"; do
  set -- $spec; j=$(submit "r24_fdiv_dev_$1_${2//,/-}" --time=05:00:00 "$D/gpu.sbatch" dev --ds "$1" --divs "$2"); DEV+=("$j"); log "DEV $spec -> $j"
done
waitjobs "${DEV[@]}"
n=$(ls "$D"/runs/dev_*/COMPLETE.json 2>/dev/null | wc -l); [ "$n" -eq 5 ] || { log "DEV incomplete ($n/5); stop"; exit 7; }
"$PY" -I -B "$D/run.py" analyze >> "$LOG" 2>&1 && log "ANALYZE done" || log "analyze failed"
