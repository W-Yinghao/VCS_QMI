#!/usr/bin/env bash
# r5 final evaluation driver (login node).  Start (PLAN_SRC = the frozen, committed plan; copied to <root>/plan.json if absent):
#   mkdir -p <root> && PLAN_SRC=<frozen plan.json> nohup setsid bash slurm/r5_final_eval_submit.sh <root> >> <root>/driver.log 2>&1 < /dev/null &
# Stop by PID only:  ps -u $USER -o pid,cmd | grep "[r]5_final_eval_submit.sh"   then   kill <pid>
# Stages with file gates (no afterok chains):
#   build (CPU) -> proposals (GPU, --partition=RTX6000PRO) -> rank shards (GPU, in parallel) -> manifest (CPU)
#   -> STOP until <root>/ALLOW_SCORING exists (the coordinator creates it only after committing the prediction hashes) -> score (CPU).
# A unit is submitted only while my normal-QOS job count is below CAP - 1 (so it stays < CAP after submitting); an empty job id stops the
# driver.  A unit whose job leaves the queue without its marker (re-checked for SETTLE seconds: NFS) is resubmitted up to MAXTRY attempts
# (proposals, rank shards, manifest), except: exit=3 in its log (a deliberate check failure) -> stop; exit=0 without a marker -> stop;
# build and score (the only stages that open the annotation sources / the labels) are not retried automatically (MAXTRY_OPEN=1).
# State (job ids, attempts) in <root>/driver_state/, so a restarted driver waits for in-flight jobs instead of submitting twice.
set -u
ROOT=${1:?usage: r5_final_eval_submit.sh <root>}; ROOT=$(cd "$ROOT" && pwd)
REPO=/home/infres/yinwang/CS_QMI/ssl_pilot; LOGS=/home/infres/yinwang/CS_QMI/slurm_logs
CAP=${CAP:-27}; POLL=${POLL:-60}; MAXTRY=${MAXTRY:-3}; GATE_POLL=${GATE_POLL:-120}; PROP_PART=${PROP_PART:-RTX6000PRO}; JN=${JN_PREFIX:-r5fe}
SETTLE=${SETTLE:-180}                  # NFS: a marker written on a compute node can appear on the login node with a delay
MAXTRY_OPEN=${MAXTRY_OPEN:-1}          # build (opens the annotation sources) and score (opens the labels): no automatic retry by default
ST=$ROOT/driver_state; mkdir -p "$ST"
log() { echo "[r5fe-driver $(date -u +%FT%TZ)] $*"; }
PLAN_SRC=${PLAN_SRC:-}                 # the frozen (committed) plan; copied byte-identically to <root>/plan.json when that does not exist yet
if [ -n "$PLAN_SRC" ]; then
  [ -f "$PLAN_SRC" ] || { log "STOP: PLAN_SRC $PLAN_SRC not found"; exit 2; }
  if [ ! -f "$ROOT/plan.json" ]; then cp "$PLAN_SRC" "$ROOT/plan.json" && log "plan installed from $PLAN_SRC"; fi
  [ "$(sha256sum < "$PLAN_SRC" | cut -d' ' -f1)" = "$(sha256sum < "$ROOT/plan.json" | cut -d' ' -f1)" ] || { log "STOP: $ROOT/plan.json differs from PLAN_SRC"; exit 2; }
fi
[ -f "$ROOT/plan.json" ] || { log "STOP: $ROOT/plan.json missing (set PLAN_SRC to the frozen plan)"; exit 2; }
if [ ! -f "$ROOT/shards.txt" ]; then
  JQ=$(command -v jq || echo /home/infres/yinwang/anaconda3/bin/jq)
  "$JQ" -r '.shards | keys[]' "$ROOT/plan.json" > "$ROOT/shards.txt" && [ -s "$ROOT/shards.txt" ] || { log "STOP: cannot list the plan's shards"; exit 2; }
fi
log "plan sha256 $(sha256sum < "$ROOT/plan.json" | cut -d' ' -f1); shards: $(tr '\n' ' ' < "$ROOT/shards.txt")"
MODE=$(grep -o '"mode": *"[a-z]*"' "$ROOT/plan.json" | head -1 | sed 's/.*"\([a-z]*\)"$/\1/')
log "start pid=$$ root=$ROOT mode=$MODE cap=$CAP maxtry=$MAXTRY proposals partition=$PROP_PART job-name prefix=$JN"
echo $$ > "$ST/driver.pid"

count_normal() {     # my normal-QOS jobs; prints nothing (and fails) when squeue is unreachable
  local out; out=$(squeue -h -u "$USER" -q normal -o %i 2>/dev/null) || return 1; echo "$out" | grep -c . ; }
wait_cap() {
  local n
  while :; do
    if n=$(count_normal) && [ "$n" -lt $((CAP - 1)) ]; then return 0; fi
    sleep "$POLL"
  done; }
job_alive() {        # 0 = pending / running / unknown (squeue unreachable); 1 = gone
  local out rc; out=$(squeue -h -j "$1" -o %T 2>&1); rc=$?
  if [ $rc -ne 0 ]; then echo "$out" | grep -qi "invalid job id" && return 1; return 0; fi
  [ -n "$out" ]; }
submit_unit() {      # unit jobname sbatch-args...
  local unit=$1 jn=$2; shift 2; wait_cap
  local jid; jid=$(sbatch --parsable --job-name="$jn" "$@" 2>>"$ST/sbatch_errors.log"); jid=${jid%%;*}
  if [ -z "$jid" ]; then log "STOP: empty job id for $unit (sbatch --job-name=$jn $*); see $ST/sbatch_errors.log"; exit 4; fi
  echo "$jid" > "$ST/$unit.jobid"; echo "$(date -u +%FT%TZ) $jid $jn" >> "$ST/$unit.attempts"; log "$unit: submitted job $jid ($jn)"; }
marker_exists() { ls "$(dirname "$1")" > /dev/null 2>&1; [ -f "$1" ]; }    # listing the directory revalidates the NFS cache
ensure_unit() {      # unit marker jobname sbatch-args... ; returns 0 once the marker exists, 1 while waiting (submits when needed)
  local unit=$1 marker=$2 jn=$3; shift 3
  marker_exists "$marker" && return 0
  local jid=""; [ -f "$ST/$unit.jobid" ] && jid=$(cat "$ST/$unit.jobid")
  if [ -n "$jid" ] && job_alive "$jid"; then return 1; fi
  if [ -n "$jid" ]; then
    local k; for k in $(seq 1 $((SETTLE / 10))); do marker_exists "$marker" && return 0; sleep 10; done
    local lf="$LOGS/${jn}_${jid}.out" last; last=$(grep "^exit=" "$lf" 2>/dev/null | tail -1)
    case "$last" in
      "exit=3 "*) log "STOP: $unit job $jid ended with a deliberate check failure (exit=3); see $lf"; exit 3;;
      "exit=0 "*) log "STOP: $unit job $jid exited 0 but $marker is still missing after ${SETTLE}s; see $lf"; exit 6;;
    esac
    local tries max=$MAXTRY; tries=$(grep -c . "$ST/$unit.attempts" 2>/dev/null || echo 0)
    case "$unit" in build|score) max=$MAXTRY_OPEN;; esac
    if [ "$tries" -ge "$max" ]; then
      log "STOP: $unit left the queue without $marker after $tries attempt(s) (last: ${last:-no exit line}; log $lf).  Inspect; to retry: rm $ST/$unit.jobid $ST/$unit.attempts and restart the driver"; exit 5
    fi
    log "$unit: job $jid left the queue without $marker (${last:-no exit line}); resubmitting (attempt $((tries + 1)) of $max)"
  fi
  submit_unit "$unit" "$jn" "$@"; return 1; }

CPU_SB=$REPO/slurm/r5_final_eval_cpu.sbatch; GPU_SB=$REPO/slurm/r5_final_eval_gpu.sbatch
until ensure_unit build "$ROOT/logs/BUILD_DONE" ${JN}_build "$CPU_SB" build --root "$ROOT"; do sleep "$POLL"; done
log "build done"
until ensure_unit proposals "$ROOT/proposals/PROPOSALS_DONE" ${JN}_prop --partition="$PROP_PART" --time=00:45:00 "$GPU_SB" proposals --root "$ROOT"; do sleep "$POLL"; done
log "proposals done"
while :; do
  all=1
  while read -r sh; do
    [ -z "$sh" ] && continue
    ensure_unit "rank_$sh" "$ROOT/sealed/predictions/shards/$sh.done" "${JN}_rank_$sh" "$GPU_SB" rank --root "$ROOT" --shard "$sh" || all=0
  done < "$ROOT/shards.txt"
  [ $all -eq 1 ] && break
  sleep "$POLL"
done
log "all rank shards done"
until ensure_unit manifest "$ROOT/sealed/predictions/MANIFEST_DONE" ${JN}_manifest --time=00:20:00 "$CPU_SB" manifest --root "$ROOT"; do sleep "$POLL"; done
log "prediction manifest written: $ROOT/sealed/predictions/MANIFEST.sha256 (sha256 $(sha256sum "$ROOT/sealed/predictions/MANIFEST.sha256" | cut -d' ' -f1))"
log "WAITING for $ROOT/ALLOW_SCORING (coordinator: commit MANIFEST.sha256 + manifest.json, then write 'manifest_sha256=<sha>' into ALLOW_SCORING)"
until [ -f "$ROOT/ALLOW_SCORING" ]; do sleep "$GATE_POLL"; done
log "ALLOW_SCORING found: $(head -c 400 "$ROOT/ALLOW_SCORING" | tr '\n' ' ')"
until ensure_unit score "$ROOT/score/SCORE_DONE" ${JN}_score "$CPU_SB" score --root "$ROOT"; do sleep "$POLL"; done
log "score done: $ROOT/score/results.json, $ROOT/score/tables.md"
exit 0
