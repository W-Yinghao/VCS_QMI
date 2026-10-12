#!/usr/bin/env bash
# R7-30 D1-30 driver (login node, ONLY after the coordinator's D1-30 staged freeze):
#   nohup setsid bash slurm/r7_d1_30_submit.sh >/dev/null 2>&1 &      — stop by PID; log slurm_logs/r7_d1_30_submit.log
# Gate 0: the frozen protocol carries the key 'd1_30_staged_freeze' (else STOP: nothing is read).
# Gate 1: D1-20 complete — selectors/lr_choice.json, then the 15 units D1-30 uses (S1-S3 × seeds 0-2 at the chosen rates, A1 / A2 × seeds 0-2): <tag>.pt and <tag>.json (file gates, give up after WAIT_H h).
# Stage 0: evaluation sets — CPU job, sbatch --wait (scripts/r7_d1_30_data.py build); gate sets/d1_30_sets.json.
# Stage 1: configuration files + unit lines (scripts/r7_d1_30_plan.py; small text files), then the 21 GPU units below CAP.  Each line carries its
#          completion marker; a unit whose job left the queue without its marker is resubmitted (every unit resumes where it stopped), at most
#          RETRIES times, then STOP.  No afterok chains (a finished dependency is purged after MinJobAge and sbatch then rejects it).
# Stage 2: readout — CPU job, sbatch --wait (scripts/r7_d1_30_score.py); gate results/d1_30_results.json.
# Every submission waits below CAP; an empty or non-numeric job id stops the driver (logged).  Overrides for dry runs: D1_ROOT, D30_ROOT,
# SUBMIT_LOG, CAP, WAIT_H, POLL, PY, PROTOCOL, LINES_FILE, RETRIES.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
CAP=${CAP:-27}; WAIT_H=${WAIT_H:-72}; POLL=${POLL:-300}; PY=${PY:-/home/infres/yinwang/CS_QMI/env/bin/python}; RETRIES=${RETRIES:-2}
D1=${D1_ROOT:-/home/infres/yinwang/CS_QMI/outputs/R7_D1}; R=${D30_ROOT:-$D1/d1_30}; P=${PROTOCOL:-reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml}
LOG=${SUBMIT_LOG:-/home/infres/yinwang/CS_QMI/slurm_logs/r7_d1_30_submit.log}; LINES=${LINES_FILE:-slurm/r7_d1_30_lines.txt}
CPUJ="--partition=CPU --gres=gpu:0 --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
waitcap() { while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -ge "$CAP" ]; do sleep "$POLL"; done; }
sub() { waitcap; ID=$(sbatch --parsable "$@") || ID=""; [[ "$ID" =~ ^[0-9]+$ ]] || { say "STOP: no job id from sbatch $*"; exit 1; }; say "$ID $*"; }
subline() { waitcap; ID=$(eval "$1") || ID=""; [[ "$ID" =~ ^[0-9]+$ ]] || { say "STOP: no job id from: $1"; exit 1; }; say "$ID $1"; }
gone() { local out rc; out=$(squeue -h -j "$1" -o %T 2>&1); rc=$?          # left the queue: listed empty, or purged ("Invalid job id");
  if [ $rc -eq 0 ]; then [ -z "$out" ]; return; fi; [[ "$out" == *"Invalid job id"* ]]; }   # any other squeue error counts as still queued
gate() { local n=0 lim=$(( WAIT_H * 3600 / (POLL > 0 ? POLL : 1) )); until [ -f "$1" ]; do
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$lim" ] && { say "STOP: $1 missing after $WAIT_H h"; exit 1; }; done; }
gatedyn() { local n=0 lim=$(( WAIT_H * 3600 / (POLL > 0 ? POLL : 1) )); until grep -q '"dynamic"' "$1" 2>/dev/null; do   # unit log rewritten with the CAL taus
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$lim" ] && { say "STOP: no dynamic taus in $1 after $WAIT_H h"; exit 1; }; done; }
say "start (D1 root $D1, D1-30 root $R, CAP $CAP, WAIT_H $WAIT_H h, POLL $POLL s, RETRIES $RETRIES)"
grep -q 'd1_30_staged_freeze' "$P" || { say "STOP: D1-30 is not frozen ($P has no d1_30_staged_freeze)"; exit 1; }
# gate 1 — D1-20 complete
gate "$D1/selectors/lr_choice.json"
UNITS=$("$PY" -I scripts/r7_d1_30_plan.py units --lr-choice "$D1/selectors/lr_choice.json") || { say "STOP: unit list"; exit 1; }
for u in $UNITS; do gate "$D1/selectors/$u.pt"; gatedyn "$D1/selectors/$u.json"; done
say "D1-20 complete: $(echo $UNITS | wc -w) units"
# stage 0 — evaluation sets (CPU, waits)
sub --wait --job-name=r7d30_build $CPUJ slurm/r7_d1_30_cpu.sbatch build
[ -f "$R/sets/d1_30_sets.json" ] || { say "STOP: build job $ID ended without sets/d1_30_sets.json"; exit 1; }
say "sets: $(tr -d '\n ' < "$R/sets/d1_30_sets.json" | cut -c1-400)"
# stage 1 — configurations, lines, GPU units with marker gates and resubmission
"$PY" -I scripts/r7_d1_30_plan.py configs --lr-choice "$D1/selectors/lr_choice.json" --out "$R/configs" >> "$LOG" 2>&1 || { say "STOP: configs"; exit 1; }
"$PY" -I scripts/r7_d1_30_plan.py lines --root "$R" --lines "$LINES" >> "$LOG" 2>&1 || { say "STOP: lines"; exit 1; }
declare -A JOB TRY CMD
while IFS= read -r line || [ -n "$line" ]; do
  [[ -z "$line" || "$line" =~ ^# ]] && continue; cmd=${line%% #MARK *}; mark=${line##* #MARK }
  CMD[$mark]=$cmd; TRY[$mark]=0
  if [ -f "$mark" ]; then say "already complete: $mark"; JOB[$mark]=""; continue; fi
  subline "$cmd"; JOB[$mark]=$ID
done < "$LINES"
n=0; lim=$(( WAIT_H * 3600 / (POLL > 0 ? POLL : 1) ))
while :; do
  left=0
  for mark in "${!CMD[@]}"; do
    [ -f "$mark" ] && continue; left=$((left + 1)); j=${JOB[$mark]}
    if [ -z "$j" ] || gone "$j"; then
      if [ "${TRY[$mark]}" -ge "$RETRIES" ]; then say "STOP: unit failed $((RETRIES + 1)) times without $mark (last job $j)"; exit 1; fi
      TRY[$mark]=$((TRY[$mark] + 1)); say "RETRY ${TRY[$mark]}/$RETRIES: job $j left the queue without $mark"; subline "${CMD[$mark]}"; JOB[$mark]=$ID
    fi
  done
  [ "$left" -eq 0 ] && break
  sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$lim" ] && { say "STOP: $left units incomplete after $WAIT_H h"; exit 1; }
done
say "stage 1 complete (${#CMD[@]} units)"
# stage 2 — readout (CPU, waits)
sub --wait --job-name=r7d30_score $CPUJ slurm/r7_d1_30_cpu.sbatch score
[ -f "$R/results/d1_30_results.json" ] || { say "STOP: score job $ID ended without results/d1_30_results.json"; exit 1; }
say "done: results/d1_30_results.json"
