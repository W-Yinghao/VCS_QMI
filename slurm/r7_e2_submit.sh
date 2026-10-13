#!/usr/bin/env bash
# r7 E2 driver (login node; stop by PID; log slurm_logs/r7_e2_submit.log).  Staged freezes, as D1-20 / D1-30 / D1-FULL / R7-40:
#   nohup setsid bash slurm/r7_e2_submit.sh teacher >/dev/null 2>&1 &   E2a, the teacher stage: needs the top-level key 'r7_e2a_staged_freeze' in the
#                                                                        frozen D1 protocol and the GPU smoke's passing probe report + replay file
#   ROWS=selectors PCA_BASIS=fitfull nohup setsid bash slurm/r7_e2_submit.sh rest >/dev/null 2>&1 &
#                                                                        cache, selectors, evaluation: needs 'r7_e2_staged_freeze'; never submits a
#                                                                        teacher unit (it waits for the teacher stage's files), so it may run while
#                                                                        the teacher driver is still running
#   ... slurm/r7_e2_submit.sh all                                        both stages in one process (teacher first, then the rest)
# File gates only (no afterok chains: a finished dependency is purged after MinJobAge).  Every submission waits until
# `squeue -h -u $USER -q normal | wc -l` < CAP (27).  An empty or non-numeric job id STOPs.  A unit whose job left the queue (listed empty, or purged:
# 'Invalid job id') without its marker after SOON_S seconds (NFS visibility) is resubmitted at most RETRIES times, then STOP; every unit resumes
# (the one-pass cache restarts).  One writer per file: a unit is resubmitted only after its previous job has left the queue, and the two stages
# never submit the same unit.  After every blocking CPU job (sbatch --wait) its output file is awaited with soon() before it is read (R7-50
# execution note r7_50_5); a check's stale output is removed before its job is resubmitted.
# teacher:  0 gates (E2a key; passing probe report and replay file from 'slurm/r7_e2_gpu.sbatch smoke')
#           1 data     CPU: verify, static reuse check, replay, E2/data, verified copies, teacher lines   -> data/data_ready.json
#           2 bundles  L40S (D40a-4), frozen R7-40a teacher                                            -> teacher/bundle_<b>.done.json
#           3 complete CPU: the frozen R7-40a completeness on E2's teacher dir                          -> teacher/r7_40a_completeness.json
# rest:     0 gates (E2 key; ROWS / PCA_BASIS / CACHE_PARTITION valid); waits for data/data_ready.json (teacher stage)
#           4 cache    one job (RTX6000PRO / H100 / L40S; independent of the teacher) + CPU cache check  -> cache/cache_d1_40.json, cache/e2_cache_check.json
#           5 waits for the teacher stage's completeness ("complete": true)
#           6 search   6 seed-0 units (RTX6000PRO,H100) -> choice (CPU, frozen rule) -> stage 2: 6 units at the chosen rates
#           7 eval     CPU check (frozen R7-40c stage 0 with E2's units / PCA / lr_choice -> E2's 50 core configs) -> run configs (ROWS) -> 12
#                      quality shards (L40S) -> CPU runcomplete -> CPU reuserows (ROWS=selectors: STOP unless it passes; it copies R7-40c's 4 grid
#                      rows) -> CPU corecomplete
# No readout: 'sbatch ... slurm/r7_e2_cpu.sbatch score' is the coordinator's step after the done line (it refuses without the freeze keys).
# Overrides (dry runs / coordinator decisions): E2_ROOT, PROTOCOL, SUBMIT_LOG, LINES_DIR, CAP, WAIT_H, POLL, PY, RETRIES, SOON_S, SOON_POLL, ROWS, PCA_BASIS, CACHE_PARTITION.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
STAGE=${1:-}
CAP=${CAP:-27}; WAIT_H=${WAIT_H:-96}; POLL=${POLL:-300}; RETRIES=${RETRIES:-2}; SOON_S=${SOON_S:-600}; SOON_POLL=${SOON_POLL:-10}
PY=${PY:-/home/infres/yinwang/CS_QMI/env/bin/python}; E2=${E2_ROOT:-/home/infres/yinwang/CS_QMI/outputs/R7_E2}
P=${PROTOCOL:-reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml}; LOG=${SUBMIT_LOG:-/home/infres/yinwang/CS_QMI/slurm_logs/r7_e2_submit.log}
LD=${LINES_DIR:-$E2/lines}; ROWS=${ROWS:-selectors}; PCA_BASIS=${PCA_BASIS:-fitfull}; CPART=${CACHE_PARTITION:-RTX6000PRO,H100,L40S}
CPUJ="--partition=CPU --gres=gpu:0 --cpus-per-task=4 --mem=32G --time=02:00:00 --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out"
TC=$E2/teacher/r7_40a_completeness.json; CC=$E2/cache/e2_cache_check.json
mkdir -p "$(dirname "$LOG")"
say() { echo "$(date -u +%FT%TZ) [e2 ${STAGE:-?}] $*" >> "$LOG"; }
stop() { say "STOP: $*"; exit 1; }
waitcap() { while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -ge "$CAP" ]; do sleep "$POLL"; done; }
subline() { waitcap; ID=$(eval "$1") || ID=""; ID=${ID%%;*}; [[ "$ID" =~ ^[0-9]+$ ]] || stop "no job id from: $1"; say "$ID $1"; }
subwait() { local out rc; waitcap; out=$(sbatch --parsable --wait "$@"); rc=$?; ID=$(echo "$out" | head -1); ID=${ID%%;*}
  [[ "$ID" =~ ^[0-9]+$ ]] || stop "no job id from sbatch --wait $*"; say "$ID (waited, rc $rc) $*"; }
gone() { local out rc; out=$(squeue -h -j "$1" -o %T 2>&1); rc=$?          # left the queue: listed empty, or purged ("Invalid job id");
  if [ $rc -eq 0 ]; then [ -z "$out" ]; return; fi; [[ "$out" == *"Invalid job id"* ]]; }   # any other squeue error counts as still queued
soon() { local n=0; until [ -f "$1" ]; do [ "$n" -ge "$SOON_S" ] && return 1; sleep "$SOON_POLL"; n=$((n + SOON_POLL)); done; return 0; }
jtrue() { [ -f "$1" ] && tr -d ' \n' < "$1" | grep -q "\"$2\":true"; }
waitfor() {     # $1 file, $2 key ('' = existence only): file gate of the other stage, up to WAIT_H hours
  local n=0 lim=$(( WAIT_H * 3600 / (POLL > 0 ? POLL : 1) )); say "waiting for $1${2:+ (\"$2\": true)}"
  until [ -f "$1" ] && { [ -z "$2" ] || jtrue "$1" "$2"; }; do sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$lim" ] && stop "$1 not ready after $WAIT_H h"; done; }
cpu_step() {    # $1 output file, $2 key that must be true, $3 job name, rest: r7_e2_cpu.sbatch arguments; skipped when the output already passes
  local f="$1" key="$2" name="$3"; shift 3
  if jtrue "$f" "$key"; then say "already passing: $f"; return 0; fi
  rm -f "$f"; subwait --job-name="$name" $CPUJ slurm/r7_e2_cpu.sbatch "$@"
  soon "$f" || stop "job $ID ($name) ended without $f"
  jtrue "$f" "$key" || stop "job $ID ($name): $f does not say \"$key\": true"; }
run_lines() {   # $1: lines file '<sbatch line> #MARK <marker>' -> submit units without marker, wait for all markers, resubmit units that left the queue
  local f="$1" line cmd mark j left n=0 lim=$(( WAIT_H * 3600 / (POLL > 0 ? POLL : 1) )); local -a ORDER=(); declare -A JOB TRY CMD
  [ -f "$f" ] || stop "lines file $f missing"
  while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue; cmd=${line%% #MARK *}; mark=${line##* #MARK }
    [[ "$cmd" == *"--export"* || "$cmd" == *"--dependency"* || "$cmd" != *"--exclude=node51,node60"* || "$cmd" != "sbatch --parsable "* ]] && stop "malformed line: $line"
    ORDER+=("$mark"); CMD[$mark]=$cmd; TRY[$mark]=0
    if [ -f "$mark" ]; then say "already complete: $mark"; JOB[$mark]=""; continue; fi
    subline "$cmd"; JOB[$mark]=$ID
  done < "$f"
  while :; do
    left=0
    for mark in "${ORDER[@]}"; do
      [ -f "$mark" ] && continue; left=$((left + 1)); j=${JOB[$mark]}
      if [ -z "$j" ] || gone "$j"; then
        soon "$mark" && continue                                                       # NFS visibility of a just-written marker
        [ "${TRY[$mark]}" -ge "$RETRIES" ] && stop "unit failed $((RETRIES + 1)) times without $mark (last job $j)"
        TRY[$mark]=$((TRY[$mark] + 1)); say "RETRY ${TRY[$mark]}/$RETRIES: job $j left the queue without $mark"; subline "${CMD[$mark]}"; JOB[$mark]=$ID
      fi
    done
    [ "$left" -eq 0 ] && break
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$lim" ] && stop "$left units of $f incomplete after $WAIT_H h"
  done
  say "complete: $f (${#ORDER[@]} units)"
}
teacher_stage() {
  grep -q '^r7_e2a_staged_freeze:' "$P" || stop "the E2 teacher stage is not frozen ($P has no top-level r7_e2a_staged_freeze)"
  grep -q '"pass": true' "$E2/smoke/probe/probe_report.json" 2>/dev/null || stop "no passing probe report in $E2/smoke/probe (run slurm/r7_e2_gpu.sbatch smoke)"
  [ -f "$E2/smoke/reuse_replay.json" ] || stop "no replay file $E2/smoke/reuse_replay.json (run slurm/r7_e2_gpu.sbatch smoke)"
  mkdir -p "$LD"
  if [ ! -f "$E2/data/data_ready.json" ]; then                                     # 1 — data (CPU, waits)
    subwait --job-name=r7e2_data $CPUJ slurm/r7_e2_cpu.sbatch data "$LD/teacher_lines.txt"
    soon "$E2/data/data_ready.json" || stop "data job $ID ended without data/data_ready.json"
  fi
  soon "$LD/teacher_lines.txt" || stop "teacher lines $LD/teacher_lines.txt missing"
  say "data: $(tr -d '\n ' < "$E2/data/data_ready.json" | cut -c1-300)"
  run_lines "$LD/teacher_lines.txt"                                                # 2 — bundles (L40S)
  cpu_step "$TC" complete r7e2_complete complete                                    # 3 — completeness (CPU, waits)
  say "teacher stage complete: $(tr -d '\n ' < "$TC" | grep -o '"present_per_role":{[^}]*}' | head -1)"
}
rest_stage() {
  grep -q '^r7_e2_staged_freeze:' "$P" || stop "E2 (cache / selectors / evaluation) is not frozen ($P has no top-level r7_e2_staged_freeze)"
  waitfor "$E2/data/data_ready.json" ""; soon "$E2/data/d1_manifest_meta.json" || stop "E2 manifest meta missing"
  mkdir -p "$LD"
  "$PY" -I scripts/r7_e2_plan.py --root "$E2" cache-lines --data-dir "$E2/data" --lines "$LD/cache_lines.txt" --basis "$PCA_BASIS" --partition "$CPART" >> "$LOG" 2>&1 || stop "cache lines"
  run_lines "$LD/cache_lines.txt"                                                  # 4 — feature cache + its check
  cpu_step "$CC" pass r7e2_cachecheck cachecheck "$PCA_BASIS"
  waitfor "$TC" complete                                                            # 5 — the teacher stage's completeness
  say "teacher complete, cache checked"
  "$PY" -I scripts/r7_e2_plan.py --root "$E2" stage1 --lines "$LD/stage1_lines.txt" >> "$LOG" 2>&1 || stop "stage-1 lines"
  run_lines "$LD/stage1_lines.txt"                                                 # 6 — learning-rate search, choice, stage 2
  if [ ! -f "$E2/selectors/lr_choice.json" ]; then
    subwait --job-name=r7e2_choose $CPUJ slurm/r7_e2_cpu.sbatch choose
    soon "$E2/selectors/lr_choice.json" || stop "choice job $ID ended without lr_choice.json"
  fi
  say "lr choice: $(tr -d '\n ' < "$E2/selectors/lr_choice.json" | cut -c1-300)"
  "$PY" -I scripts/r7_e2_plan.py --root "$E2" stage2 --selectors-dir "$E2/selectors" --lines "$LD/stage2_lines.txt" >> "$LOG" 2>&1 || stop "stage-2 lines"
  run_lines "$LD/stage2_lines.txt"
  for t in $("$PY" -I scripts/r7_e2_plan.py units --lr-choice "$E2/selectors/lr_choice.json"); do soon "$E2/selectors/$t.done.json" || stop "unit $t has no marker"; done
  say "selectors complete (12 units)"
  cpu_step "$E2/eval/check/check.json" pass r7e2_check check                     # 7 — evaluation
  for b in gqa_dev textvqa_val chartqa_val; do soon "$E2/eval/configs/configs_$b.txt" || stop "core configs for $b missing"; done
  "$PY" -I scripts/r7_e2_plan.py --root "$E2" run-configs --configs-dir "$E2/eval/configs" --out "$E2/eval/run" --rows "$ROWS" >> "$LOG" 2>&1 || stop "run configs"
  "$PY" -I scripts/r7_e2_plan.py --root "$E2" eval-lines --lines "$LD/eval_lines.txt" >> "$LOG" 2>&1 || stop "eval lines"
  run_lines "$LD/eval_lines.txt"
  cpu_step "$E2/eval/run/completeness.json" complete r7e2_runcomplete runcomplete
  local RR=$E2/eval/reuse/reuse_check.json
  if ! jtrue "$RR" pass; then
    rm -f "$RR"; subwait --job-name=r7e2_reuserows $CPUJ slurm/r7_e2_cpu.sbatch reuserows "$ROWS"
    soon "$RR" || stop "reuserows job $ID ended without $RR"
    if ! jtrue "$RR" pass; then
      [ "$ROWS" = selectors ] && stop "R7-40c's full / grid rows failed the reuse check ($RR): coordinator decision (e.g. ROWS=core in a fresh eval dir)"
      say "reuse check not passing (ROWS=core: descriptive only; E2's own grid rows are used)"
    fi
  fi
  cpu_step "$E2/eval/completeness.json" complete r7e2_corecomplete corecomplete
  say "done: E2 evaluation complete (readout: sbatch --job-name=r7e2_score $CPUJ slurm/r7_e2_cpu.sbatch score — the coordinator's step)"
}
main() {        # the whole file is parsed before main runs (last line), so editing this file cannot disturb a running driver
  case "$STAGE" in teacher|rest|all) ;; *) STAGE=""; stop "usage: slurm/r7_e2_submit.sh teacher | rest | all";; esac
  say "start pid $$ stage $STAGE (E2 $E2, ROWS $ROWS, PCA_BASIS $PCA_BASIS, CACHE_PARTITION $CPART, CAP $CAP, WAIT_H $WAIT_H h, POLL $POLL s, RETRIES $RETRIES, SOON_S $SOON_S s)"
  if [ "$STAGE" != teacher ]; then                                                 # validate the rest's settings before anything is submitted
    case "$ROWS" in selectors|core) ;; *) stop "bad ROWS '$ROWS' (selectors | core)";; esac
    case "$PCA_BASIS" in fitfull|frozen10k) ;; *) stop "bad PCA_BASIS '$PCA_BASIS' (fitfull | frozen10k)";; esac
    case "$CPART" in *[!A-Za-z0-9,]*|"") stop "bad CACHE_PARTITION '$CPART'";; esac
    grep -q '^r7_e2_staged_freeze:' "$P" || stop "E2 (cache / selectors / evaluation) is not frozen ($P has no top-level r7_e2_staged_freeze)"
  fi
  [ "$STAGE" = rest ] || teacher_stage
  if [ "$STAGE" = teacher ]; then say "done: E2 teacher stage complete (the rest: slurm/r7_e2_submit.sh rest, after the E2 freeze)"; return 0; fi
  rest_stage
}
main "$@"; exit $?
