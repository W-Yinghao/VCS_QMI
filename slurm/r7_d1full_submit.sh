#!/usr/bin/env bash
# R7-30 D1-FULL driver (login node, ONLY after the coordinator's freeze: key 'd1_full_staged_freeze' in the frozen D1 protocol):
#   nohup setsid bash slurm/r7_d1full_submit.sh >/dev/null 2>&1 &      — stop by PID; log slurm_logs/r7_d1full_submit.log
# File gates only (no afterok chains: a finished dependency is purged after MinJobAge and sbatch then rejects it).  Every submission waits until
# `squeue -h -u $USER -q normal | wc -l` < CAP (27).  An empty or non-numeric job id STOPs.  A unit whose job left the queue (listed empty or
# 'Invalid job id') without its marker is resubmitted — every unit resumes where it stopped — at most RETRIES times, then STOP.  One writer per
# file: a unit is resubmitted only after its previous job has left the queue.
#   0 gates   : freeze key in the protocol; the GPU smoke's probe report passes (the teacher bundles' launch gate)
#   1 data    : CPU job (--wait): build if absent, verify, freeze the D1-FULL manifest, reuse check + copy, plan, teacher lines -> data/data_ready.json
#   2 teacher : bundle lines (LLaVA env, H100 / L40S)                                                                  -> teacher/bundle_<id>.ok.json
#   3 complete: CPU job (--wait): every CAL / FIT shard complete                                                       -> teacher/d1full_completeness.json
#   4 cache   : one LLaVA-env job, PCA_BASIS=fitfull (default) | frozen                                                 -> cache/cache_d1_full.json
#   5 search  : 6 seed-0 units (RTX6000PRO / H100)                                                                      -> selectors/<tag>.complete.json
#   6 choice  : CPU job (--wait), frozen D1-20 rule                                                                     -> selectors/lr_choice.json (visibility wait, E4)
#   7 stage 2 : 12 units at the chosen rates                                                                            -> selectors/<tag>.complete.json
#   8 eval    : D1-30 frozen + its sets built -> selector-row configs + 12 quality shards + 3 dynamic-row timing units on nodesumo01 (TIMING=1,
#               coordinator: required), then DEV risk (DEVRISK=1) once D1-30's own DEV-risk output exists (D1-30 reads the sealed DEV records first);
#               outputs in DF/d1_30 in the frozen D1-30 format (reference for the official-test phase)
# No readout here: scripts/r7_d1full_score.py is run by the coordinator after the D1-30 readout.
# Overrides (dry runs / coordinator decisions): DF_ROOT, D1_ROOT, PROTOCOL, SUBMIT_LOG, LINES_DIR, CAP, WAIT_H, POLL, PY, RETRIES, PCA_BASIS, DEVRISK, TIMING.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
CAP=${CAP:-27}; WAIT_H=${WAIT_H:-96}; POLL=${POLL:-300}; PY=${PY:-/home/infres/yinwang/CS_QMI/env/bin/python}; RETRIES=${RETRIES:-2}
DF=${DF_ROOT:-/home/infres/yinwang/CS_QMI/outputs/R7_D1_FULL}; D1=${D1_ROOT:-/home/infres/yinwang/CS_QMI/outputs/R7_D1}
P=${PROTOCOL:-reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml}; LOG=${SUBMIT_LOG:-/home/infres/yinwang/CS_QMI/slurm_logs/r7_d1full_submit.log}
LD=${LINES_DIR:-$DF/lines}; PCA_BASIS=${PCA_BASIS:-fitfull}; DEVRISK=${DEVRISK:-1}; TIMING=${TIMING:-1}
CPUJ="--partition=CPU --gres=gpu:0 --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out"
mkdir -p "$LD"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
stop() { say "STOP: $*"; exit 1; }
waitcap() { while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -ge "$CAP" ]; do sleep "$POLL"; done; }
subline() { waitcap; ID=$(eval "$1") || ID=""; [[ "$ID" =~ ^[0-9]+$ ]] || stop "no job id from: $1"; say "$ID $1"; }
subwait() {     # CPU step, sbatch --wait: STOP on a missing job id or a failed job
  local out rc; waitcap; out=$(sbatch --parsable --wait "$@"); rc=$?; ID=$(echo "$out" | head -1)
  [[ "$ID" =~ ^[0-9]+$ ]] || stop "no job id from sbatch $*"; say "$ID (waited, rc $rc) $*"; [ "$rc" -eq 0 ] || stop "job $ID failed (rc $rc): $*"; }
gone() { local out rc; out=$(squeue -h -j "$1" -o %T 2>&1); rc=$?          # left the queue: listed empty, or purged ("Invalid job id");
  if [ $rc -eq 0 ]; then [ -z "$out" ]; return; fi; [[ "$out" == *"Invalid job id"* ]]; }   # any other squeue error counts as still queued
waitfile() { local n=0 lim=$(( ${2:-600} / (POLL > 0 ? POLL : 1) + 1 )); until [ -f "$1" ]; do
    [ "$n" -ge "$lim" ] && return 1; sleep "$POLL"; n=$((n + 1)); done; return 0; }
run_lines() {   # $1: lines file '<sbatch line> #MARK <marker>' -> submit units without marker, wait for all markers, resubmit units that left the queue
  local f="$1" line cmd mark j left n=0 lim=$(( WAIT_H * 3600 / (POLL > 0 ? POLL : 1) )); declare -A JOB TRY CMD
  [ -f "$f" ] || stop "lines file $f missing"
  while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue; cmd=${line%% #MARK *}; mark=${line##* #MARK }
    [[ "$cmd" == *"--export"* || "$cmd" != *"--exclude=node51,node52,node60"* ]] && stop "malformed line: $line"
    CMD[$mark]=$cmd; TRY[$mark]=0
    if [ -f "$mark" ]; then say "already complete: $mark"; JOB[$mark]=""; continue; fi
    subline "$cmd"; JOB[$mark]=$ID
  done < "$f"
  while :; do
    left=0
    for mark in "${!CMD[@]}"; do
      [ -f "$mark" ] && continue; left=$((left + 1)); j=${JOB[$mark]}
      if [ -z "$j" ] || gone "$j"; then
        waitfile "$mark" 60 && continue                                                 # NFS visibility of a just-written marker (E4)
        [ "${TRY[$mark]}" -ge "$RETRIES" ] && stop "unit failed $((RETRIES + 1)) times without $mark (last job $j)"
        TRY[$mark]=$((TRY[$mark] + 1)); say "RETRY ${TRY[$mark]}/$RETRIES: job $j left the queue without $mark"; subline "${CMD[$mark]}"; JOB[$mark]=$ID
      fi
    done
    [ "$left" -eq 0 ] && break
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$lim" ] && stop "$left units of $f incomplete after $WAIT_H h"
  done
  say "complete: $f (${#CMD[@]} units)"
}
say "start (DF $DF, D1 $D1, CAP $CAP, WAIT_H $WAIT_H h, POLL $POLL s, RETRIES $RETRIES, PCA_BASIS $PCA_BASIS, DEVRISK $DEVRISK, TIMING $TIMING)"
# 0 — gates
grep -q 'd1_full_staged_freeze' "$P" || stop "D1-FULL is not frozen ($P has no d1_full_staged_freeze)"
[[ "$PCA_BASIS" == fitfull || "$PCA_BASIS" == frozen ]] || stop "PCA_BASIS must be fitfull or frozen"
grep -q '"pass": true' "$DF/smoke/probe/probe_report.json" 2>/dev/null || stop "no passing probe report in $DF/smoke/probe (run the GPU smoke)"
# 1 — data (CPU, waits)
if [ ! -f "$DF/data/data_ready.json" ]; then
  subwait --job-name=r7dF_data $CPUJ slurm/r7_d1full_cpu.sbatch data --lines "$LD/teacher_lines.txt"
  waitfile "$DF/data/data_ready.json" 600 || stop "data job $ID ended without data/data_ready.json"
fi
say "data: $(tr -d '\n ' < "$DF/data/data_ready.json" | cut -c1-300)"
# 2 — teacher bundles
run_lines "$LD/teacher_lines.txt"
# 3 — completeness (CPU, waits)
if ! grep -q '"complete": true' "$DF/teacher/d1full_completeness.json" 2>/dev/null; then
  subwait --job-name=r7dF_complete $CPUJ slurm/r7_d1full_cpu.sbatch complete
  waitfile "$DF/teacher/d1full_completeness.json" 600; grep -q '"complete": true' "$DF/teacher/d1full_completeness.json" 2>/dev/null || stop "teacher incomplete"
fi
say "teacher complete"
# 4 — feature cache
echo "sbatch --parsable --job-name=r7dF_cache --partition=H100,L40S --exclude=node51,node52,node60 --time=06:00:00 slurm/r7_d1full_gpu.sbatch cache --pca-basis $PCA_BASIS #MARK $DF/cache/cache_d1_full.json" > "$LD/cache_lines.txt"
run_lines "$LD/cache_lines.txt"
grep -q "\"basis\": \"$PCA_BASIS\"" "$DF/cache/cache_d1_full.json" || stop "cache basis differs from PCA_BASIS=$PCA_BASIS"
# 5 — learning-rate search, seed 0
"$PY" -I scripts/r7_d1full_plan.py stage1 --selectors-dir "$DF/selectors" --lines "$LD/stage1_lines.txt" >> "$LOG" 2>&1 || stop "stage-1 lines"
run_lines "$LD/stage1_lines.txt"
# 6 — choice (CPU, waits; frozen D1-20 rule)
if [ ! -f "$DF/selectors/lr_choice.json" ]; then
  subwait --job-name=r7dF_choose $CPUJ slurm/r7_d1full_cpu.sbatch choose
  waitfile "$DF/selectors/lr_choice.json" 900 || stop "choice job $ID ended without lr_choice.json"
fi
say "lr choice: $(tr -d '\n ' < "$DF/selectors/lr_choice.json" | cut -c1-300)"
# 7 — stage 2
"$PY" -I scripts/r7_d1full_plan.py stage2 --selectors-dir "$DF/selectors" --lines "$LD/stage2_lines.txt" >> "$LOG" 2>&1 || stop "stage-2 lines"
run_lines "$LD/stage2_lines.txt"
# 8 — evaluation: frozen D1-30 runner on the D1-30 sets (built by the D1-30 driver), D1-FULL checkpoints, separate output root
grep -q 'd1_30_staged_freeze' "$P" || stop "D1-30 not frozen"
waitfile "$D1/d1_30/sets/d1_30_sets.json" $(( WAIT_H * 3600 )) || stop "D1-30 evaluation sets missing after $WAIT_H h"
"$PY" -I scripts/r7_d1full_plan.py eval-configs --lr-choice "$DF/selectors/lr_choice.json" --out "$DF/d1_30/configs" >> "$LOG" 2>&1 || stop "eval configs"
EXTRA="--no-devrisk"; [ "$TIMING" = 1 ] || EXTRA="$EXTRA --no-timing"
"$PY" -I scripts/r7_d1full_plan.py eval-lines --root "$DF/d1_30" --lines "$LD/eval_lines.txt" $EXTRA >> "$LOG" 2>&1 || stop "eval lines"
run_lines "$LD/eval_lines.txt"
if [ "$DEVRISK" = 1 ]; then   # D1-30's DEV-risk unit is the first read of the sealed D1-10 DEV records: the D1-FULL one waits for its output (existence only)
  waitfile "$D1/d1_30/devrisk/d1_30_devrisk.json" $(( WAIT_H * 3600 )) || stop "D1-30 DEV-risk output absent after $WAIT_H h"
  "$PY" -I scripts/r7_d1full_plan.py eval-lines --root "$DF/d1_30" --lines "$LD/devrisk_lines.txt" --devrisk-only >> "$LOG" 2>&1 || stop "devrisk line"
  run_lines "$LD/devrisk_lines.txt"
fi
say "done: D1-FULL evaluation complete (readout: scripts/r7_d1full_score.py, by the coordinator after the D1-30 readout)"
