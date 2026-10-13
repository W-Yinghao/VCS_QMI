#!/usr/bin/env bash
# R7-50 phase Q driver (login node): Qwen2.5-VL-7B-Instruct rows on GQA testdev-balanced and ChartQA test under the frozen addendum
# reports/R7/R7_50_PHASE_Q_ADDENDUM_DRAFT_20261012.yaml (status frozen_20261013; Q1–Q8), procedure A4 of the frozen R7-50 protocol unchanged.
# The frozen phase-L driver (slurm/r7_50_submit.sh) still refuses PHASE=Q; this is phase Q's own driver.  Run ONLY after the coordinator has
# committed the phase-Q code:
#   cd /home/infres/yinwang/CS_QMI/ssl_pilot && nohup setsid bash slurm/r7_50_q_submit.sh >/dev/null 2>&1 &
#   stop by PID (kill <pid>, never pkill -f); log slurm_logs/r7_50_submit_Q.log (its first line carries the PID)
# Gate 0: the R7-50 protocol ($PROTOCOL: 'status: frozen' + a top-level 'r7_50_freeze_<yyyymmdd>:' key), the addendum ($ADDENDUM: a line
#   'status: frozen…') and R7-40c's freeze key 'r7_40c_staged_freeze' in $PROTOCOL_40C; the protocol, the addendum, the phase-Q code, the
#   R7-50 / R7-40c / D1 / D1-30 modules it imports and R7_DATA_CLOSED_SHA256.csv committed and unchanged (git); else STOP before anything is read.
# Gate 1 (phase Q runs 'after R7-40 complete'): phase L's sealed inputs (phase Q has no build step: no second read of a test file, Q3), R7-40b's
#   lr_choice.json and its 9 units (.pt / .json), R7-40c complete = the done line of slurm_logs/r7_d1_40c_submit.log AND eval_40c/completeness.json
#   'complete': true AND the 8 gqa_dev / chartqa_val reference markers, R7-40c's single nodesumo01 timing bundle (eval_40c/timing/
#   timing_bundle.done.json + the gqa_dev / chartqa_val timing files, Q2 / Q7) and the R7-40a B* report.  File gates, WAIT_H h each, then STOP.
# Integrity gate (Q3 / Q6): integrity-prepare (CPU) -> one GPU unit per R7-40c reference shard (GQA 6 + ChartQA 2, 3 rows each), pinned to the
#   reference marker's GPU class -> integrity-compare (CPU); STOP unless the top-level 'pass' of integrity_Q/integrity.json is true.
# Stage 0: configs_Q + lines_Q (CPU).  Stage 1: the 8 generation units (GQA testdev 6, ChartQA test 2; L40S).  Stage 2: manifest_Q (CPU).
# Stage 3: STOP AND WAIT for $ROOT/ALLOW_SCORING_Q (the coordinator commits a copy of the manifest, then writes manifest_sha256=, commit= and
#   manifest_copy= into it, as phase L; the scorer checks the committed bytes).  Stage 4: score (CPU; the only phase-Q job that opens the
#   sealed labels), once.
# A unit is resubmitted only after its job has left the queue (listed empty, or purged) with its marker still missing after SOON_S s (NFS),
# at most RETRIES times, then STOP; no afterok chains; submissions only while fewer than CAP (27) normal-QOS jobs are queued; an empty or
# non-numeric job id stops the driver; positional wrapper arguments only (no --export).
# Dry-run overrides (stubbed sbatch / squeue): R750_DRYRUN=1 (skips the git check, refuses the production root), R750_ROOT, D1_ROOT, R40C_ROOT,
# R40C_LOG, SUBMIT_LOG, CAP, WAIT_H, WAIT_SCORE_H, POLL, SOON_S, SOON_POLL, PY, PROTOCOL, ADDENDUM, PROTOCOL_40C, RETRIES; PHASE must be Q (default).
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
PHASE=${PHASE:-Q}; PROD_ROOT=/home/infres/yinwang/CS_QMI/outputs/R7_50
ROOT=${R750_ROOT:-$PROD_ROOT}; CAP=${CAP:-27}; WAIT_H=${WAIT_H:-96}; WAIT_SCORE_H=${WAIT_SCORE_H:-336}; POLL=${POLL:-300}; RETRIES=${RETRIES:-2}
SOON_S=${SOON_S:-120}; SOON_POLL=${SOON_POLL:-10}
PY=${PY:-/home/infres/yinwang/CS_QMI/env/bin/python}; D1=${D1_ROOT:-/home/infres/yinwang/CS_QMI/outputs/R7_D1}; R40=$D1/r7_40; RC=${R40C_ROOT:-$R40/eval_40c}
P=${PROTOCOL:-reports/R7/R7_50_PROTOCOL_FROZEN_20261012.yaml}; ADD=${ADDENDUM:-reports/R7/R7_50_PHASE_Q_ADDENDUM_DRAFT_20261012.yaml}
P40=${PROTOCOL_40C:-reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml}; R40LOG=${R40C_LOG:-/home/infres/yinwang/CS_QMI/slurm_logs/r7_d1_40c_submit.log}
LOG=${SUBMIT_LOG:-/home/infres/yinwang/CS_QMI/slurm_logs/r7_50_submit_Q.log}
CPUJ="--partition=CPU --gres=gpu:0 --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out"
say() { echo "$(date -u +%FT%TZ) [Q] $*" >> "$LOG"; }
lim() { echo $(( $1 * 3600 / (POLL > 0 ? POLL : 1) )); }
jtrue() { "$PY" -I -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get(sys.argv[2]) is True else 1)' "$1" "$2" 2>/dev/null; }   # top-level key only
waitcap() { while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -ge "$CAP" ]; do sleep "$POLL"; done; }
subwait() { waitcap; local out rc; out=$(sbatch --parsable --wait "$@"); rc=$?; ID=$(echo "$out" | head -1); ID=${ID%%;*}   # blocks until the job ends
  [[ "$ID" =~ ^[0-9]+$ ]] || { say "STOP: no job id from sbatch --wait $*"; exit 1; }; say "$ID $* (exit $rc)"
  [ "$rc" -eq 0 ] || { say "STOP: job $ID exited $rc"; exit 1; }; }
subline() { waitcap; ID=$(eval "$1") || ID=""; ID=${ID%%;*}; [[ "$ID" =~ ^[0-9]+$ ]] || { say "STOP: no job id from: $1"; exit 1; }; say "$ID $1"; }
gone() { local out rc; out=$(squeue -h -j "$1" -o %T 2>&1); rc=$?          # left the queue: listed empty, or purged ("Invalid job id");
  if [ $rc -eq 0 ]; then [ -z "$out" ]; return; fi; [[ "$out" == *"Invalid job id"* ]]; }   # any other squeue error counts as still queued
soon() { local n=0; until [ -f "$1" ]; do [ "$n" -ge "$SOON_S" ] && return 1; sleep "$SOON_POLL"; n=$((n + SOON_POLL)); done; return 0; }
gate() { local n=0 l; l=$(lim "$WAIT_H"); until [ -f "$1" ]; do
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$l" ] && { say "STOP: $1 missing after $WAIT_H h"; exit 1; }; done; }
gatelog() { local n=0 l; l=$(lim "$WAIT_H"); until grep -qF "$2" "$1" 2>/dev/null; do
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$l" ] && { say "STOP: '$2' not in $1 after $WAIT_H h"; exit 1; }; done; }
gatetrue() { local n=0 l; l=$(lim "$WAIT_H"); until jtrue "$1" "$2"; do
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$l" ] && { say "STOP: no '$2': true in $1 after $WAIT_H h"; exit 1; }; done; }
run_lines() {   # $1 lines file, $2 label: every unit without its marker is submitted; one whose job left the queue without it is resubmitted
  local file=$1 label=$2 line cmd mark n=0 l left j; declare -A JOB=() TRY=() CMD=(); local -a ORDER=()
  while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue; cmd=${line%% #MARK *}; mark=${line##* #MARK }
    ORDER+=("$mark"); CMD[$mark]=$cmd; TRY[$mark]=0; JOB[$mark]=""
    if [ -f "$mark" ]; then say "$label: already complete: $mark"; continue; fi
    subline "$cmd"; JOB[$mark]=$ID
  done < "$file"
  l=$(lim "$WAIT_H")
  while :; do
    left=0
    for mark in "${ORDER[@]}"; do
      [ -f "$mark" ] && continue; left=$((left + 1)); j=${JOB[$mark]}
      if [ -z "$j" ] || gone "$j"; then
        soon "$mark" && continue
        if [ "${TRY[$mark]}" -ge "$RETRIES" ]; then say "STOP: $label unit failed $((RETRIES + 1)) times without $mark (last job $j)"; exit 1; fi
        TRY[$mark]=$((TRY[$mark] + 1)); say "RETRY ${TRY[$mark]}/$RETRIES: job $j left the queue without $mark"; subline "${CMD[$mark]}"; JOB[$mark]=$ID
      fi
    done
    [ "$left" -eq 0 ] && break
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$l" ] && { say "STOP: $left $label units incomplete after $WAIT_H h"; exit 1; }
  done
  say "$label complete (${#ORDER[@]} units)"; }
say "start pid $$ (phase $PHASE, root $ROOT, protocol $P, addendum $ADD, R7-40c root $RC, CAP $CAP, WAIT_H $WAIT_H h, WAIT_SCORE_H $WAIT_SCORE_H h, POLL $POLL s, RETRIES $RETRIES)"
[ "$PHASE" = Q ] || { say "STOP: this driver runs phase Q only (phases L / F: slurm/r7_50_submit.sh)"; exit 1; }
# gate 0 — frozen protocol and addendum, R7-40c frozen, committed code
[ -f "$P" ] && grep -qE '^status:[[:space:]]*frozen' "$P" && grep -qE '^r7_50_freeze_[0-9]{8}[[:space:]]*:' "$P" \
  || { say "STOP: $P is not a frozen R7-50 protocol (status: frozen + top-level r7_50_freeze_<yyyymmdd>: key); nothing read"; exit 1; }
[ -f "$ADD" ] && grep -qE '^status:[[:space:]]*frozen' "$ADD" || { say "STOP: $ADD is not the frozen phase-Q addendum (status: frozen…); nothing read"; exit 1; }
grep -q 'r7_40c_staged_freeze' "$P40" 2>/dev/null || { say "STOP: R7-40c is not frozen ($P40 has no r7_40c_staged_freeze); nothing read"; exit 1; }
if [ "${R750_DRYRUN:-0}" = 1 ]; then
  [ "$ROOT" != "$PROD_ROOT" ] || { say "STOP: a dry run refuses the production root"; exit 1; }
  say "DRY RUN: git check skipped"
else
  FILES="$P $ADD scripts/r7_50_q.py scripts/r7_50_test.py slurm/r7_50_q_cpu.sbatch slurm/r7_50_q_gpu.sbatch slurm/r7_50_q_submit.sh
         scripts/r7_d1_40c_run.py scripts/r7_d1_40c_plan.py scripts/r7_d1_40c_score.py scripts/r7_d1_40c_timing.py scripts/r7_d1_40_host.py
         scripts/r7_d1_40_masks.py scripts/r7_d1_40_selectors.py scripts/r7_d1_plan20.py scripts/r7_d1_30_data.py scripts/r7_d1_30_plan.py
         scripts/r7_d1_30_run.py scripts/r7_d1_30_score.py scripts/r7_d1_30_sparsevlm.py scripts/r7_d1_30_eval.py scripts/r7_d1_30_timing.py
         scripts/r7_d1_data.py scripts/r7_d1_masks.py scripts/r7_d1_selectors.py scripts/r7_d1_teacher.py scripts/r7_vl_common.py
         reports/R7/R7_DATA_CLOSED_SHA256.csv"
  git ls-files --error-unmatch $FILES >/dev/null 2>&1 || { say "STOP: protocol / addendum / code not committed"; exit 1; }
  git diff --quiet HEAD -- $FILES || { say "STOP: protocol / addendum / code differ from HEAD"; exit 1; }
  say "gate 0 ok: git HEAD $(git rev-parse HEAD)"
fi
# gate 1 — phase L's sealed inputs (no test file is read in phase Q), R7-40b's units, R7-40c complete incl. its timing bundle
jtrue "$ROOT/sealed/inputs/build_summary.json" ok || { say "STOP: phase Q needs phase L's sealed inputs ($ROOT/sealed/inputs/build_summary.json ok); no test file is read here"; exit 1; }
gate "$R40/selectors/lr_choice.json"
UNITS=$("$PY" -I scripts/r7_d1_40c_plan.py units --lr-choice "$R40/selectors/lr_choice.json" --used) || { say "STOP: R7-40b unit list"; exit 1; }
[ "$(echo $UNITS | wc -w)" -eq 9 ] || { say "STOP: expected 9 R7-40b units, got: $UNITS"; exit 1; }
for u in $UNITS; do gate "$R40/selectors/$u.pt"; gate "$R40/selectors/$u.json"; done
say "R7-40b units present: $UNITS"
gatelog "$R40LOG" "done: R7-40c generation complete"; gatetrue "$RC/completeness.json" complete
for i in 0 1 2 3 4 5; do gate "$RC/pred/gqa_dev/done_shard${i}of6.json"; done
for i in 0 1; do gate "$RC/pred/chartqa_val/done_shard${i}of2.json"; done
# OPEN: the timing bundle is gated here, with R7-40c's completion ('after R7-40 complete'), not just before the manifest as phase L's execution
# note r7_50_1 moved D1-30's timing; manifest_Q hashes these files, and score_Q re-checks them.
gate "$RC/timing/timing_bundle.done.json"; for b in gqa_dev chartqa_val; do gate "$RC/timing/$b/timing_$b.json"; done; gate "$R40/timing/timing_report.json"
say "R7-40c complete (done line, completeness, 8 reference markers) and its timing bundle present (Q2 / Q7)"
# full-scale integrity gate (non-test data; before any test generation)
I=$ROOT/integrity_Q
if jtrue "$I/integrity.json" pass; then
  say "integrity gate already passed ($I/integrity.json)"
else
  [ -f "$I/lines.txt" ] || subwait --job-name=r7_50_iprep_Q $CPUJ slurm/r7_50_q_cpu.sbatch integrity-prepare "$ROOT"
  soon "$I/lines.txt" || { say "STOP: $I/lines.txt missing"; exit 1; }          # NFS: the CPU job wrote it on another node
  run_lines "$I/lines.txt" "integrity"
  waitcap; out=$(sbatch --parsable --wait --job-name=r7_50_icmp_Q $CPUJ slurm/r7_50_q_cpu.sbatch integrity-compare "$ROOT"); rc=$?
  soon "$I/integrity.json"; jtrue "$I/integrity.json" pass || { say "STOP: full-scale integrity gate FAILED (job ${out%%;*}, exit $rc; see $I/integrity.json); nothing generated on test"; exit 1; }
  say "integrity gate passed (job ${out%%;*}; $I/integrity.json)"
fi
# stage 0 — configs_Q + lines_Q (no test file opened; the sealed inputs are phase L's)
[ -f "$ROOT/lines_Q.txt" ] || subwait --job-name=r7_50_configs_Q $CPUJ slurm/r7_50_q_cpu.sbatch configs "$ROOT"
soon "$ROOT/lines_Q.txt" || { say "STOP: $ROOT/lines_Q.txt missing"; exit 1; }
say "stage 0 complete: configs_Q, $(grep -vc '^#' "$ROOT/lines_Q.txt") unit lines"
# stage 1 — generation
run_lines "$ROOT/lines_Q.txt" "generation Q"
# stage 2 — manifest (CPU, waits)
M=$ROOT/manifest_Q
[ -f "$M/predictions_manifest.sha256" ] || subwait --job-name=r7_50_manifest_Q $CPUJ slurm/r7_50_q_cpu.sbatch manifest "$ROOT"
soon "$M/predictions_manifest.json" && soon "$M/predictions_manifest.sha256" || { say "STOP: manifest_Q missing"; exit 1; }
MSHA=$(cut -d' ' -f1 "$M/predictions_manifest.sha256")
say "stage 2 complete: manifest_Q sha256 $MSHA.  WAITING for $ROOT/ALLOW_SCORING_Q (coordinator: commit a copy of the manifest, then write manifest_sha256=$MSHA, commit=<that commit> and manifest_copy=<repo path of the copy> into it)"
# stage 3 — wait for the coordinator's gate file
AL=$ROOT/ALLOW_SCORING_Q; n=0; l=$(lim "$WAIT_SCORE_H")
until [ -f "$AL" ]; do sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$l" ] && { say "STOP: no $AL after $WAIT_SCORE_H h"; exit 1; }; done
grep -q "^manifest_sha256=$MSHA\$" "$AL" || { say "STOP: $AL does not carry manifest_sha256=$MSHA; labels stay sealed"; exit 1; }
say "stage 3: $(basename "$AL") present ($(grep -v '^manifest_sha256=' "$AL" | tr '\n' ' ' | cut -c1-200))"
# stage 4 — score (CPU, waits): the only phase-Q job that opens the sealed labels; scored once
[ -f "$ROOT/results_Q/r7_50_results.json" ] && { say "done already: $ROOT/results_Q/r7_50_results.json exists (scored once; a re-verification is a manual step with a new out dir)"; exit 0; }
subwait --job-name=r7_50_score_Q $CPUJ slurm/r7_50_q_cpu.sbatch score "$ROOT"
soon "$ROOT/results_Q/r7_50_results.json" || { say "STOP: score job $ID ended without results_Q/r7_50_results.json"; exit 1; }
say "done: $ROOT/results_Q/r7_50_results.json and .md"
