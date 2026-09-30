#!/usr/bin/env bash
# Submission orchestrator (owner 2026-09-30: "除了imagenet的先不提交，后续都可以提交，你设置好提交程序").
# Every POLL seconds, for each row of slurm/orchestrator_units.tsv (unit <TAB> frozen-prereg glob <TAB> lines file <TAB> extra condition command):
#   submit the unit's lines (slurm/feed_lines.sbatch, normal QOS, 30-job cap respected) exactly once, when
#   (1) a FROZEN pre-registration matching the glob exists (freeze-before-compute is enforced here, never bypassed),
#   (2) the lines file exists and is non-empty, (3) the extra condition command exits 0 ("true" if none).
# Marker slurm/orchestrator_state/<unit>.submitted prevents re-submission; the log is slurm_logs/orchestrator.log.
# Never submits to the runfill QOS (owner 2026-09-30) and never touches ImageNet units (not in the table).
set -u
cd /home/infres/yinwang/CS_QMI/ssl_pilot
POLL=${POLL:-300}; STATE=slurm/orchestrator_state; mkdir -p "$STATE"
LOG=/home/infres/yinwang/CS_QMI/slurm_logs/orchestrator.log
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
log "orchestrator start pid $$"
while :; do
  while IFS=$'\t' read -r unit prereg lines cond; do
    [[ -z "$unit" || "$unit" =~ ^# ]] && continue
    [ -f "$STATE/$unit.submitted" ] && continue
    ls $prereg >/dev/null 2>&1 || continue
    [ -s "$lines" ] || continue
    grep -q "qos=runfill" "$lines" && { log "$unit: lines file requests runfill — refused"; touch "$STATE/$unit.refused"; continue; }
    bash -c "${cond:-true}" >/dev/null 2>&1 || continue
    log "$unit: conditions met ($prereg, $lines) — submitting"
    LINES_FILE="$lines" STAGE="$unit (orchestrator)" CAP=29 bash slurm/feed_lines.sbatch >> "$LOG" 2>&1 && touch "$STATE/$unit.submitted" && log "$unit: submitted"
  done < slurm/orchestrator_units.tsv
  [ -f "$STATE/STOP" ] && { log "STOP file found — exiting"; exit 0; }
  sleep "$POLL"
done
