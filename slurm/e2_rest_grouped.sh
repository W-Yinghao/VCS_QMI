#!/usr/bin/env bash
# r7 E2 — the frozen 'rest' path of slurm/r7_e2_submit.sh (feature cache, selectors, evaluation) with GROUPED unit jobs (owner decisions
# 2026-10-13: RTX PRO 6000 cards are hard to get and longer jobs are wanted; per-unit L40S jobs are grouped).  Packaging only: the same stages,
# gates, keys, CPU steps, frozen planner calls (scripts/r7_e2_plan.py writes the same lines files), markers and checks as the frozen rest path;
# every unit runs its FROZEN per-unit wrapper with its frozen arguments inside a group job (slurm/e2_unit_group.sbatch).  No unit, row, rule or
# analysis changes.
#   ROWS=selectors PCA_BASIS=fitfull nohup setsid bash slurm/e2_rest_grouped.sh >/dev/null 2>&1 &   — stop by PID; log slurm_logs/e2_rest_grouped.log
# Groups (one GPU each, units one after the other): selector units 3 per job in lines order (stage 1: 2 jobs, stage 2: 2 jobs; RTX6000PRO,H100,
# --time 12:00:00 ≈ 3 × 3 units of 1.1–1.45 h); quality shards one job per benchmark (L40S; GQA 6 shards --time 10:00:00, TextVQA 4 shards
# 16:00:00, ChartQA 2 shards 04:30:00 ≈ 3 × the expected group time with ROWS=selectors or core; ≤ 24 h).  The feature cache stays the frozen
# single-unit line (job r7e2c_cache).  A group is complete only when all its unit markers exist; a group whose job left the queue (listed empty
# or purged) with a marker still missing after SOON_S seconds is resubmitted — the group wrapper skips completed units, every unit resumes — at
# most RETRIES times, then STOP.
# Switch-over safety (no unit ever has two writers): before any submission (cache, group or CPU step) the driver looks for a queued / running job
# of mine with the same name — the frozen per-unit names (r7e2c_cache, r7e2s_<tag>, r7e2q_<bench>_<i>), the CPU-step names (r7e2_<step>) or the
# group name — and waits for it to leave the queue instead of submitting; it refuses to start while a frozen rest driver
# ('r7_e2_submit.sh rest|all') is running.
# CAP 20 (backfill) on `squeue -h -u $USER -q normal | wc -l` before every submission; normal QOS; no --export; no afterok chains; positional
# arguments only; soon() after every blocking CPU job; an empty or non-numeric job id STOPs.  No readout: 'slurm/r7_e2_cpu.sbatch score' stays the
# coordinator's step after the done line.
# Overrides (dry runs / coordinator decisions): E2_ROOT, PROTOCOL, SUBMIT_LOG, LINES_DIR, CAP, WAIT_H, POLL, PY, RETRIES, SOON_S, SOON_POLL, ROWS,
# PCA_BASIS, CACHE_PARTITION, FROZEN_DRIVER_PATTERN.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
CAP=${CAP:-20}; WAIT_H=${WAIT_H:-96}; POLL=${POLL:-300}; RETRIES=${RETRIES:-2}; SOON_S=${SOON_S:-600}; SOON_POLL=${SOON_POLL:-10}
PY=${PY:-/home/infres/yinwang/CS_QMI/env/bin/python}; E2=${E2_ROOT:-/home/infres/yinwang/CS_QMI/outputs/R7_E2}
P=${PROTOCOL:-reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml}; LOG=${SUBMIT_LOG:-/home/infres/yinwang/CS_QMI/slurm_logs/e2_rest_grouped.log}
LD=${LINES_DIR:-$E2/lines}; ROWS=${ROWS:-selectors}; PCA_BASIS=${PCA_BASIS:-fitfull}; CPART=${CACHE_PARTITION:-RTX6000PRO,H100,L40S}
FDP=${FROZEN_DRIVER_PATTERN:-r7_e2_submit\.sh (rest|all)}
CPUJ="--partition=CPU --gres=gpu:0 --cpus-per-task=4 --mem=32G --time=02:00:00 --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out"
TC=$E2/teacher/r7_40a_completeness.json; CC=$E2/cache/e2_cache_check.json; GW=slurm/e2_unit_group.sbatch; EXCL="--exclude=node51,node60"
mkdir -p "$(dirname "$LOG")"
say() { echo "$(date -u +%FT%TZ) [e2 grouped] $*" >> "$LOG"; }
stop() { say "STOP: $*"; exit 1; }
waitcap() { while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -ge "$CAP" ]; do sleep "$POLL"; done; }
subline() { waitcap; ID=$(eval "$1") || ID=""; ID=${ID%%;*}; [[ "$ID" =~ ^[0-9]+$ ]] || stop "no job id from: $1"; say "$ID $1"; }
subwait() { local out rc; waitcap; out=$(sbatch --parsable --wait "$@"); rc=$?; ID=$(echo "$out" | head -1); ID=${ID%%;*}
  [[ "$ID" =~ ^[0-9]+$ ]] || stop "no job id from sbatch --wait $*"; say "$ID (waited, rc $rc) $*"; }
gone() { local out rc; out=$(squeue -h -j "$1" -o %T 2>&1); rc=$?          # left the queue: listed empty, or purged ("Invalid job id");
  if [ $rc -eq 0 ]; then [ -z "$out" ]; return; fi; [[ "$out" == *"Invalid job id"* ]]; }   # any other squeue error counts as still queued
queued() { squeue -h -u "$USER" -o "%i %j" 2>/dev/null | awk -v n=" $* " 'index(n, " " $2 " ") {print $1}' | tr '\n' ' ' | sed 's/ *$//'; }   # my jobs with these names
soon() { local n=0; until [ -f "$1" ]; do [ "$n" -ge "$SOON_S" ] && return 1; sleep "$SOON_POLL"; n=$((n + SOON_POLL)); done; return 0; }
allm() { local m; for m in $1; do [ -f "$m" ] || return 1; done; return 0; }
soonall() { local n=0; until allm "$1"; do [ "$n" -ge "$SOON_S" ] && return 1; sleep "$SOON_POLL"; n=$((n + SOON_POLL)); done; return 0; }
missing() { local m o=""; for m in $1; do [ -f "$m" ] || o="$o $(basename "$m")"; done; echo "${o# }"; }
jtrue() { [ -f "$1" ] && tr -d ' \n' < "$1" | grep -q "\"$2\":true"; }
waitfor() {     # $1 file, $2 key ('' = existence only): file gate of another stage, up to WAIT_H hours (as the frozen rest path)
  local n=0 lim=$(( WAIT_H * 3600 / (POLL > 0 ? POLL : 1) )); say "waiting for $1${2:+ (\"$2\": true)}"
  until [ -f "$1" ] && { [ -z "$2" ] || jtrue "$1" "$2"; }; do sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$lim" ] && stop "$1 not ready after $WAIT_H h"; done; }
cpu_run() {     # $1 job name, rest: slurm/r7_e2_cpu.sbatch arguments; a queued job of that name (switch-over) is waited for instead of a new one
  local name=$1 q j; shift; q=$(queued "$name")
  if [ -n "$q" ]; then say "waiting for queued $name job(s) $q"; for j in $q; do until gone "$j"; do sleep "$POLL"; done; done; ID=${q%% *}; return 0; fi
  subwait --job-name="$name" $CPUJ slurm/r7_e2_cpu.sbatch "$@"; }
cpu_step() {    # $1 output file, $2 key that must be true, $3 job name, rest: r7_e2_cpu.sbatch arguments; skipped when the output already passes
  local f="$1" key="$2" name="$3"; shift 3
  if jtrue "$f" "$key"; then say "already passing: $f"; return 0; fi
  [ -n "$(queued "$name")" ] || rm -f "$f"                                             # a stale output is removed unless a job is writing it
  cpu_run "$name" "$@"
  soon "$f" || stop "job $ID ($name) ended without $f"
  jtrue "$f" "$key" || stop "job $ID ($name): $f does not say \"$key\": true"; }
checkline() { [[ "$1" == *"--export"* || "$1" == *"--dependency"* || "$1" != *"$EXCL"* || "$1" != "sbatch --parsable "* ]] && stop "malformed line: $1"; return 0; }
jobname() { local x; x=$(grep -oE -- '--job-name=[^ ]+' <<< "$1" | head -1); echo "${x#--job-name=}"; }
run_units() {   # frozen run_lines semantics for single-unit lines ('<sbatch line> #MARK <marker>') + the switch-over adoption by job name
  local f="$1" line cmd mark j q left n=0 lim=$(( WAIT_H * 3600 / (POLL > 0 ? POLL : 1) )); local -a ORDER=(); declare -A JOB TRY CMD SUB
  [ -f "$f" ] || stop "lines file $f missing"
  while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue; cmd=${line%% #MARK *}; mark=${line##* #MARK }; checkline "$cmd"
    ORDER+=("$mark"); CMD[$mark]=$cmd; TRY[$mark]=0; SUB[$mark]=0; JOB[$mark]=""
  done < "$f"
  while :; do
    left=0
    for mark in "${ORDER[@]}"; do
      [ -f "$mark" ] && continue; left=$((left + 1)); j=${JOB[$mark]}
      if [ -z "$j" ] || gone "$j"; then
        [ -n "$j" ] && soon "$mark" && continue                                        # NFS visibility of a just-written marker
        q=$(queued "$(jobname "${CMD[$mark]}")")
        if [ -n "$q" ]; then JOB[$mark]=${q%% *}; say "waiting for queued job(s) $q ($(jobname "${CMD[$mark]}"))"; continue; fi
        if [ "${SUB[$mark]}" -eq 1 ]; then
          [ "${TRY[$mark]}" -ge "$RETRIES" ] && stop "unit failed $((RETRIES + 1)) times without $mark (last job $j)"
          TRY[$mark]=$((TRY[$mark] + 1)); say "RETRY ${TRY[$mark]}/$RETRIES: job $j left the queue without $mark"
        fi
        subline "${CMD[$mark]}"; JOB[$mark]=$ID; SUB[$mark]=1
      fi
    done
    [ "$left" -eq 0 ] && break
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$lim" ] && stop "$left units of $f incomplete after $WAIT_H h"
  done
  say "complete: $f (${#ORDER[@]} units)"
}
make_groups() { # $1 frozen lines file, $2 group lines file, $3 mode sel (3 units per group, in lines order) | bench (one group per benchmark), $4 name prefix
  local f="$1" out="$2" mode="$3" pre="$4" line cmd i k key gname part tm; local -a N=() M=() PA=() K=() G=()
  [ -f "$f" ] || stop "lines file $f missing"
  while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue; cmd=${line%% #MARK *}; checkline "$cmd"
    N+=("$(jobname "$cmd")"); M+=("${line##* #MARK }"); PA+=("$(grep -oE -- '--partition=[^ ]+' <<< "$cmd" | head -1)")
    if [ "$mode" = bench ]; then K+=("$(awk '{for (i = 1; i < NF; i++) if ($i == "run") {print $(i + 1); exit}}' <<< "$cmd")"); else K+=(""); fi
    [ -n "${N[-1]}" ] && [ -n "${PA[-1]}" ] && [[ "$cmd" == *" slurm/r7_e2_sel.sbatch train "* || "$cmd" == *" slurm/r7_e2_gpu.sbatch run "* ]] || stop "not a frozen selector / quality line: $line"
  done < "$f"
  [ "${#N[@]}" -gt 0 ] || stop "no unit in $f"
  for ((i = 0; i < ${#N[@]}; i++)); do                                               # group index of every unit
    if [ "$mode" = sel ]; then G+=("$((i / 3 + 1))"); else G+=("${K[i]}"); fi
  done
  { echo "# r7 E2 grouped units from $f ($mode): <sbatch line> #MARKS <unit markers>  (slurm/e2_rest_grouped.sh; packaging only)"
    for gname in $(printf '%s\n' "${G[@]}" | awk '!seen[$0]++'); do
      local units="" marks="" p=""
      for ((i = 0; i < ${#N[@]}; i++)); do
        [ "${G[i]}" = "$gname" ] || continue
        [ -z "$p" ] && p=${PA[i]}; [ "${PA[i]}" = "$p" ] || stop "units of group $gname have different partitions"
        units="$units ${N[i]}"; marks="$marks ${M[i]}"
      done
      case "$mode:$gname" in sel:*) tm=12:00:00;; bench:gqa_dev) tm=10:00:00;; bench:textvqa_val) tm=16:00:00;; bench:chartqa_val) tm=04:30:00;; *) stop "no time limit for $mode group $gname";; esac
      echo "sbatch --parsable --job-name=${pre}${gname} $p $EXCL --time=$tm $GW $f${units} #MARKS${marks}"
    done; } > "$out.tmp" && mv "$out.tmp" "$out"
  say "groups: $(grep -vc '^#' "$out") jobs for ${#N[@]} units -> $out"
}
run_groups() {  # $1 group lines '<sbatch line> #MARKS <marker> ...': a group is complete when all its markers exist; resubmitted (at most RETRIES) if its job left the queue without them
  local f="$1" line cmd g j q us left n=0 lim=$(( WAIT_H * 3600 / (POLL > 0 ? POLL : 1) )); local -a ORDER=(); declare -A JOB TRY CMD MK SUB UN
  [ -f "$f" ] || stop "group lines file $f missing"
  while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue; cmd=${line%% #MARKS *}; checkline "$cmd"; g=$(jobname "$cmd")
    us=${cmd#* $GW }; us=${us#* }                                                     # the unit names (after the lines file)
    ORDER+=("$g"); CMD[$g]=$cmd; MK[$g]=${line#* #MARKS }; UN[$g]=$us; TRY[$g]=0; SUB[$g]=0; JOB[$g]=""
  done < "$f"
  while :; do
    left=0
    for g in "${ORDER[@]}"; do
      allm "${MK[$g]}" && continue; left=$((left + 1)); j=${JOB[$g]}
      if [ -z "$j" ] || gone "$j"; then
        [ -n "$j" ] && soonall "${MK[$g]}" && continue                                  # NFS visibility of just-written markers
        q=$(queued "$g" ${UN[$g]})                                                       # a live job of this group or of one of its units
        if [ -n "$q" ]; then JOB[$g]=${q%% *}; say "waiting for queued job(s) $q (group $g or its units)"; continue; fi
        if [ "${SUB[$g]}" -eq 1 ]; then
          [ "${TRY[$g]}" -ge "$RETRIES" ] && stop "group $g failed $((RETRIES + 1)) times (missing: $(missing "${MK[$g]}"); last job $j)"
          TRY[$g]=$((TRY[$g] + 1)); say "RETRY ${TRY[$g]}/$RETRIES: group job $j left the queue without $(missing "${MK[$g]}")"
        fi
        subline "${CMD[$g]}"; JOB[$g]=$ID; SUB[$g]=1
      fi
    done
    [ "$left" -eq 0 ] && break
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$lim" ] && stop "$left groups of $f incomplete after $WAIT_H h"
  done
  say "complete: $f (${#ORDER[@]} groups)"
}
main() {        # the whole file is parsed before main runs (last line)
  say "start pid $$ (E2 $E2, ROWS $ROWS, PCA_BASIS $PCA_BASIS, CACHE_PARTITION $CPART, CAP $CAP, WAIT_H $WAIT_H h, POLL $POLL s, RETRIES $RETRIES, SOON_S $SOON_S s)"
  case "$ROWS" in selectors|core) ;; *) stop "bad ROWS '$ROWS' (selectors | core)";; esac          # the frozen rest path's settings and key
  case "$PCA_BASIS" in fitfull|frozen10k) ;; *) stop "bad PCA_BASIS '$PCA_BASIS' (fitfull | frozen10k)";; esac
  case "$CPART" in *[!A-Za-z0-9,]*|"") stop "bad CACHE_PARTITION '$CPART'";; esac
  grep -q '^r7_e2_staged_freeze:' "$P" || stop "E2 (cache / selectors / evaluation) is not frozen ($P has no top-level r7_e2_staged_freeze)"
  local fd; fd=$(pgrep -u "$USER" -f "$FDP" | tr '\n' ' '); [ -z "$fd" ] || stop "a frozen rest driver is still running (pid $fd): stop it first (one driver per stage)"
  waitfor "$E2/data/data_ready.json" ""; soon "$E2/data/d1_manifest_meta.json" || stop "E2 manifest meta missing"
  mkdir -p "$LD"
  "$PY" -I scripts/r7_e2_plan.py --root "$E2" cache-lines --data-dir "$E2/data" --lines "$LD/cache_lines.txt" --basis "$PCA_BASIS" --partition "$CPART" >> "$LOG" 2>&1 || stop "cache lines"
  run_units "$LD/cache_lines.txt"                                                     # 4 — the feature cache (already submitted: adopted or skipped)
  cpu_step "$CC" pass r7e2_cachecheck cachecheck "$PCA_BASIS"
  waitfor "$TC" complete                                                               # 5 — the teacher stage's completeness
  say "teacher complete, cache checked"
  "$PY" -I scripts/r7_e2_plan.py --root "$E2" stage1 --lines "$LD/stage1_lines.txt" >> "$LOG" 2>&1 || stop "stage-1 lines"
  make_groups "$LD/stage1_lines.txt" "$LD/groups_stage1.txt" sel r7e2g_s1_; run_groups "$LD/groups_stage1.txt"           # 6 — search
  if [ ! -f "$E2/selectors/lr_choice.json" ]; then                                    #     choice (CPU, frozen rule)
    cpu_run r7e2_choose choose; soon "$E2/selectors/lr_choice.json" || stop "choice job $ID ended without lr_choice.json"
  fi
  say "lr choice: $(tr -d '\n ' < "$E2/selectors/lr_choice.json" | cut -c1-300)"
  "$PY" -I scripts/r7_e2_plan.py --root "$E2" stage2 --selectors-dir "$E2/selectors" --lines "$LD/stage2_lines.txt" >> "$LOG" 2>&1 || stop "stage-2 lines"
  make_groups "$LD/stage2_lines.txt" "$LD/groups_stage2.txt" sel r7e2g_s2_; run_groups "$LD/groups_stage2.txt"           #     stage 2
  local t; for t in $("$PY" -I scripts/r7_e2_plan.py units --lr-choice "$E2/selectors/lr_choice.json"); do soon "$E2/selectors/$t.done.json" || stop "unit $t has no marker"; done
  say "selectors complete (12 units)"
  cpu_step "$E2/eval/check/check.json" pass r7e2_check check                         # 7 — evaluation
  local b; for b in gqa_dev textvqa_val chartqa_val; do soon "$E2/eval/configs/configs_$b.txt" || stop "core configs for $b missing"; done
  "$PY" -I scripts/r7_e2_plan.py --root "$E2" run-configs --configs-dir "$E2/eval/configs" --out "$E2/eval/run" --rows "$ROWS" >> "$LOG" 2>&1 || stop "run configs"
  "$PY" -I scripts/r7_e2_plan.py --root "$E2" eval-lines --lines "$LD/eval_lines.txt" >> "$LOG" 2>&1 || stop "eval lines"
  make_groups "$LD/eval_lines.txt" "$LD/groups_eval.txt" bench r7e2g_q_; run_groups "$LD/groups_eval.txt"
  cpu_step "$E2/eval/run/completeness.json" complete r7e2_runcomplete runcomplete
  local RR=$E2/eval/reuse/reuse_check.json
  if ! jtrue "$RR" pass; then
    [ -n "$(queued r7e2_reuserows)" ] || rm -f "$RR"; cpu_run r7e2_reuserows reuserows "$ROWS"
    soon "$RR" || stop "reuserows job $ID ended without $RR"
    if ! jtrue "$RR" pass; then
      [ "$ROWS" = selectors ] && stop "R7-40c's full / grid rows failed the reuse check ($RR): coordinator decision (e.g. ROWS=core in a fresh eval dir)"
      say "reuse check not passing (ROWS=core: descriptive only; E2's own grid rows are used)"
    fi
  fi
  cpu_step "$E2/eval/completeness.json" complete r7e2_corecomplete corecomplete
  say "done: E2 evaluation complete (readout: sbatch --job-name=r7e2_score $CPUJ slurm/r7_e2_cpu.sbatch score — the coordinator's step)"
}
main "$@"; exit $?
