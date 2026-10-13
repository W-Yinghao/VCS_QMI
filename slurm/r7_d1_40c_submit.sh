#!/usr/bin/env bash
# R7-40c driver (login node, ONLY after the coordinator's staged freeze of R7-40c):
#   QUALITY_PARTITION=L40S ROWS=core BSTAR=report DEV=1 TIMING=1 nohup setsid bash slurm/r7_d1_40c_submit.sh >/dev/null 2>&1 &   — stop by PID; log
#   slurm_logs/r7_d1_40c_submit.log
# gate 0: the frozen D1 protocol carries 'r7_40c_staged_freeze' (else STOP: nothing is read or submitted).
# gate 1: R7-40b complete — selectors/lr_choice.json and the markers <tag>.done.json of its 12 units (6 seed-0 learning-rate-search units + 6
#         seed-1 / seed-2 units at the chosen rates); file gates, give up after WAIT_H h.
# stage 0: CPU check (sbatch --wait): the frozen D1-30 sets against their build summary, every image present, merged grids within the
#          selectors' 64 × 64 tables, the 9 units' files and CAL taus, configuration files for ROWS (core | static | dynamic | all) and
#          configs/rows.json with sha256s -> STOP unless check/check.json says pass.
# stage 1: unit lines (scripts/r7_d1_40c_plan.py lines): quality shards (QUALITY_PARTITION); the batch-1 official full-row anchors (decision 15:
#          the full row through the official HF generate, one unit per benchmark on QUALITY_PARTITION); ONE timing bundle on nodesumo01 (owner rule
#          2026-10-13: every nodesumo01 unit in one job, back-to-back, one marker, retried as a whole) = Qwen's B* (BSTAR=in_bundle: the frozen
#          R7-40a procedure runs first inside the bundle unless its report exists — the driver STOPs if the R7-40a timing job 'r7d40_timing' is
#          still queued, so B* is measured once; BSTAR=report: the bundle waits for that job's report) + the three benchmarks; the two sealed DEV
#          teacher bundles (frozen R7-40a code, --allow-dev, L40S); the DEV risk secondary gated on both bundle markers.
#          A unit is submitted only while its marker is missing and all its #GATE files exist, and
#          resubmitted only after its job has left the queue (listed empty, or purged: 'Invalid job id') with the marker still missing after
#          SOON_S seconds (NFS visibility) — at most RETRIES times, then STOP.  No two jobs ever write the same files; no afterok chains.
# stage 2: CPU completeness (sbatch --wait): every configured row and the anchor cover each benchmark's items, every shard marker -> STOP unless
#          complete.
# Then the driver stops: scoring is the coordinator's step (sbatch ... slurm/r7_d1_40c_cpu.sbatch score), which refuses without the freeze key.
# Every submission waits below CAP (normal QOS, as the frozen drivers); an empty or non-numeric job id stops the driver.  DEV=0 / TIMING=0 drop
# those units (only if the freeze says so); BSTAR=report (default; coordinator decision 14) | in_bundle.  Overrides for dry runs: D1_ROOT, R40_ROOT, R40C_ROOT, SUBMIT_LOG,
# CAP, WAIT_H, POLL, PY, PROTOCOL, RETRIES, SOON_S, SOON_POLL, LINES_DIR, TIMING_REPORT.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
CAP=${CAP:-27}; WAIT_H=${WAIT_H:-96}; POLL=${POLL:-300}; RETRIES=${RETRIES:-2}; SOON_S=${SOON_S:-600}; SOON_POLL=${SOON_POLL:-10}
PY=${PY:-/home/infres/yinwang/CS_QMI/env/bin/python}; D1=${D1_ROOT:-/home/infres/yinwang/CS_QMI/outputs/R7_D1}; R=${R40_ROOT:-$D1/r7_40}
RC=${R40C_ROOT:-$R/eval_40c}; P=${PROTOCOL:-reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml}; LD=${LINES_DIR:-slurm}; ROWS=${ROWS:-core}
QPART=${QUALITY_PARTITION:-L40S}; DEV=${DEV:-1}; TIMING=${TIMING:-1}; TREPORT=${TIMING_REPORT:-$R/timing/timing_report.json}; BSTAR=${BSTAR:-report}
LOG=${SUBMIT_LOG:-/home/infres/yinwang/CS_QMI/slurm_logs/r7_d1_40c_submit.log}
CPUJ="--partition=CPU --gres=gpu:0 --cpus-per-task=4 --mem=32G --time=01:00:00 --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out"
say() { echo "$(date -u +%FT%TZ) [40c] $*" >> "$LOG"; }
waitcap() { while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -ge "$CAP" ]; do sleep "$POLL"; done; }
subwait() { waitcap; ID=$(sbatch --parsable --wait "$@"); local rc=$?; ID=${ID%%;*}
  [[ "$ID" =~ ^[0-9]+$ ]] || { say "STOP: no job id from sbatch --wait $*"; exit 1; }; say "$ID (rc $rc) $*"; }
subline() { waitcap; ID=$(eval "$1") || ID=""; ID=${ID%%;*}; [[ "$ID" =~ ^[0-9]+$ ]] || { say "STOP: no job id from: $1"; exit 1; }; say "$ID $1"; }
gone() { local out rc; out=$(squeue -h -j "$1" -o %T 2>&1); rc=$?          # left the queue: listed empty, or purged ("Invalid job id");
  if [ $rc -eq 0 ]; then [ -z "$out" ]; return; fi; [[ "$out" == *"Invalid job id"* ]]; }   # any other squeue error counts as still queued
soon() { local n=0; until [ -f "$1" ]; do [ "$n" -ge "$SOON_S" ] && return 1; sleep "$SOON_POLL"; n=$((n + SOON_POLL)); done; return 0; }
gate() { local n=0 lim=$(( WAIT_H * 3600 / (POLL > 0 ? POLL : 1) )); until [ -f "$1" ]; do
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$lim" ] && { say "STOP: $1 missing after $WAIT_H h"; exit 1; }; done; }
case "$QPART" in *[!A-Za-z0-9,]*|"") say "STOP: bad QUALITY_PARTITION '$QPART'"; exit 1;; esac
case "$ROWS" in core|static|dynamic|all) ;; *) say "STOP: bad ROWS '$ROWS'"; exit 1;; esac
case "$BSTAR" in in_bundle|report) ;; *) say "STOP: bad BSTAR '$BSTAR'"; exit 1;; esac
bstar_guard() {   # B* is determined once: with BSTAR=in_bundle and no report yet, the R7-40a timing job must not be queued (two writers of one report)
  [ "$TIMING" = "0" ] || [ "$BSTAR" != "in_bundle" ] || [ -f "$TREPORT" ] && return 0
  local q; q=$(squeue -h -u "$USER" -n r7d40_timing -o %i 2>/dev/null)
  [ -z "$q" ] || { say "STOP: BSTAR=in_bundle but the R7-40a timing job r7d40_timing ($q) is still queued: cancel it (coordinator) so B* is measured once inside the bundle, or relaunch with BSTAR=report"; exit 1; }; }
run_units() {   # $1: lines file '<sbatch line> #MARK <marker> [#GATE <file> ...]'; submission and retries in file order
  local line cmd rest mark g ready j left n=0 lim=$(( WAIT_H * 3600 / (POLL > 0 ? POLL : 1) )); local -a ORDER=(); declare -A JOB TRY CMD GATES SUB
  while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    cmd=${line%% #MARK *}; rest=${line##* #MARK }; mark=${rest%% #GATE *}; GATES[$mark]=""; [[ "$rest" == *" #GATE "* ]] && GATES[$mark]=${rest#* #GATE }
    ORDER+=("$mark"); CMD[$mark]=$cmd; TRY[$mark]=0; JOB[$mark]=""; SUB[$mark]=0
    if [ -f "$mark" ]; then say "already complete: $mark"; SUB[$mark]=1; fi
  done < "$1"
  while :; do
    left=0
    for mark in "${ORDER[@]}"; do
      [ -f "$mark" ] && continue; left=$((left + 1))
      if [ "${SUB[$mark]}" -eq 0 ]; then                                          # first submission once every gate file exists
        ready=1; for g in ${GATES[$mark]}; do [ -f "$g" ] || ready=0; done
        if [ "$ready" -eq 1 ]; then subline "${CMD[$mark]}"; JOB[$mark]=$ID; SUB[$mark]=1; fi
        continue
      fi
      j=${JOB[$mark]}
      if [ -z "$j" ] || gone "$j"; then
        soon "$mark" && continue
        if [ "${TRY[$mark]}" -ge "$RETRIES" ]; then say "STOP: unit failed $((RETRIES + 1)) times without $mark (last job $j)"; exit 1; fi
        TRY[$mark]=$((TRY[$mark] + 1)); say "RETRY ${TRY[$mark]}/$RETRIES: job $j left the queue without $mark"; subline "${CMD[$mark]}"; JOB[$mark]=$ID
      fi
    done
    [ "$left" -eq 0 ] && break
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$lim" ] && { say "STOP: $left units incomplete (or gated) after $WAIT_H h"; exit 1; }
  done
}
say "start pid $$ (D1 root $D1, R7-40 root $R, R7-40c root $RC, rows $ROWS, quality partition $QPART, DEV $DEV, TIMING $TIMING, BSTAR $BSTAR, CAP $CAP, WAIT_H $WAIT_H h, POLL $POLL s, RETRIES $RETRIES)"
grep -q 'r7_40c_staged_freeze' "$P" || { say "STOP: R7-40c is not frozen ($P has no r7_40c_staged_freeze)"; exit 1; }
bstar_guard
gate "$R/selectors/lr_choice.json"
TAGS=$("$PY" -I scripts/r7_d1_40c_plan.py units --lr-choice "$R/selectors/lr_choice.json") || { say "STOP: unit list"; exit 1; }
[ "$(echo $TAGS | wc -w)" -eq 12 ] || { say "STOP: expected 12 R7-40b units, got: $TAGS"; exit 1; }
for t in $TAGS; do gate "$R/selectors/$t.done.json"; done
say "gate 1: R7-40b complete (lr_choice.json + 12 unit markers)"
subwait --job-name=r7d40c_check $CPUJ slurm/r7_d1_40c_cpu.sbatch check "$ROWS"
{ soon "$RC/check/check.json" && grep -q '"pass": true' "$RC/check/check.json"; } || { say "STOP: stage-0 check failed or missing ($RC/check/check.json)"; exit 1; }
say "stage 0 complete: check passes ($(tr -d '\n ' < "$RC/check/check.json" | grep -o '"configs":{[^}]*}' | head -1))"
EXTRA=""; [ "$DEV" = "0" ] && EXTRA="$EXTRA --no-dev"; [ "$TIMING" = "0" ] && EXTRA="$EXTRA --no-timing"
"$PY" -I scripts/r7_d1_40c_plan.py lines --root "$RC" --lines "$LD/r7_d1_40c_lines.txt" --quality-partition "$QPART" --r40 "$R" --timing-report "$TREPORT" \
  --bstar "$BSTAR" $EXTRA >> "$LOG" 2>&1 || { say "STOP: planner failed"; exit 1; }
bstar_guard; run_units "$LD/r7_d1_40c_lines.txt"; say "stage 1 complete: quality shards, official anchors, the timing bundle, DEV units"
subwait --job-name=r7d40c_complete $CPUJ slurm/r7_d1_40c_cpu.sbatch complete
{ soon "$RC/completeness.json" && grep -q '"complete": true' "$RC/completeness.json"; } || { say "STOP: completeness check failed ($RC/completeness.json)"; exit 1; }
say "done: R7-40c generation complete; scoring is the coordinator's step: sbatch --job-name=r7d40c_score $CPUJ slurm/r7_d1_40c_cpu.sbatch score"
