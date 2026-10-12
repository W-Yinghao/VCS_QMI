#!/usr/bin/env bash
# R7-40 driver (login node, ONLY after the coordinator's staged freeze of the stage it runs):
#   nohup setsid bash slurm/r7_d1_40_submit.sh 40a >/dev/null 2>&1 &     (later: ... 40b)  — stop by PID; log slurm_logs/r7_d1_40_submit.log
# 40a  gate 0: the protocol carries 'r7_40a_staged_freeze' and the D1 manifest is frozen (else STOP: nothing runs).
#      stage 0: the launch probe (GPU; sentinels S0–S7 under the frozen S4 rule S4_RULE) -> STOP unless its report passes.
#      stage 1: the 12 CAL / FIT teacher bundles of the frozen shard plan (CAL first; DEV bundles are not part of 40a).
#      stage 2: completeness (CPU, waits) -> STOP unless complete; the CAL readout (CPU, waits).  Then stop: R7-40b needs its own freeze.
# 40b  gate 0: the protocol carries 'r7_40b_staged_freeze' and 40a is complete (teacher/r7_40a_completeness.json "complete": true).
#      stage 0: the feature cache (GPU).  stage 1: 6 learning-rate-search units.  choice: CPU job (waits), frozen OPEN-6 rule -> lr_choice.json.
#      stage 2: 6 units (seeds 1-2 at the chosen rates).
# Units: one sbatch line per unit with its completion marker (#MARK).  A unit is submitted only while its marker is missing, and resubmitted only
# after its job has left the queue (listed empty, or purged: 'Invalid job id') and the marker is still missing after SOON_S seconds (NFS
# visibility, D1-20 execution note E4) — at most RETRIES times, then STOP.  So no two jobs ever write the same files, and there are no afterok
# chains (a finished dependency is purged after MinJobAge and sbatch then rejects it).  Every submission waits below CAP (normal QOS); an empty or
# non-numeric job id stops the driver.  Launch parameters fixed by the freeze (record them there): S4_RULE (min | rate:<r>), S1_RULE (r720 |
# same_gpu), PROBE_PARTITION, TEACHER_PARTITION, CACHE_PARTITION, SEL_PARTITION (comma lists of RTX6000PRO / H100 / L40S; used only in the sbatch
# --partition option, never in --export).  Overrides for dry runs: D1_ROOT, R40_ROOT, SUBMIT_LOG, CAP, WAIT_H, POLL, PY, PROTOCOL, RETRIES,
# SOON_S, SOON_POLL, LINES_DIR.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
STAGE=${1:-}; CAP=${CAP:-27}; WAIT_H=${WAIT_H:-72}; POLL=${POLL:-300}; RETRIES=${RETRIES:-2}; SOON_S=${SOON_S:-600}; SOON_POLL=${SOON_POLL:-10}
PY=${PY:-/home/infres/yinwang/CS_QMI/env/bin/python}; D1=${D1_ROOT:-/home/infres/yinwang/CS_QMI/outputs/R7_D1}; R=${R40_ROOT:-$D1/r7_40}
P=${PROTOCOL:-reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml}; S4_RULE=${S4_RULE:-min}; S1_RULE=${S1_RULE:-r720}; LD=${LINES_DIR:-slurm}
PROBE_PART=${PROBE_PARTITION:-RTX6000PRO,H100,L40S}; TEACHER_PART=${TEACHER_PARTITION:-RTX6000PRO,H100,L40S}; CACHE_PART=${CACHE_PARTITION:-RTX6000PRO,H100,L40S}
SEL_PART=${SEL_PARTITION:-RTX6000PRO,H100}
LOG=${SUBMIT_LOG:-/home/infres/yinwang/CS_QMI/slurm_logs/r7_d1_40_submit.log}
CPUJ="--partition=CPU --gres=gpu:0 --cpus-per-task=4 --mem=48G --time=02:00:00 --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out"
say() { echo "$(date -u +%FT%TZ) [$STAGE] $*" >> "$LOG"; }
waitcap() { while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -ge "$CAP" ]; do sleep "$POLL"; done; }
subwait() { waitcap; ID=$(sbatch --parsable --wait "$@"); local rc=$?; ID=${ID%%;*}
  [[ "$ID" =~ ^[0-9]+$ ]] || { say "STOP: no job id from sbatch --wait $*"; exit 1; }; say "$ID (rc $rc) $*"; }
subline() { waitcap; ID=$(eval "$1") || ID=""; ID=${ID%%;*}; [[ "$ID" =~ ^[0-9]+$ ]] || { say "STOP: no job id from: $1"; exit 1; }; say "$ID $1"; }
gone() { local out rc; out=$(squeue -h -j "$1" -o %T 2>&1); rc=$?          # left the queue: listed empty, or purged ("Invalid job id");
  if [ $rc -eq 0 ]; then [ -z "$out" ]; return; fi; [[ "$out" == *"Invalid job id"* ]]; }   # any other squeue error counts as still queued
soon() { local n=0; until [ -f "$1" ]; do [ "$n" -ge "$SOON_S" ] && return 1; sleep "$SOON_POLL"; n=$((n + SOON_POLL)); done; return 0; }
plan() { "$PY" -I scripts/r7_d1_40_plan.py --root "$R" "$@" >> "$LOG" 2>&1 || { say "STOP: planner failed: $*"; exit 1; }; }
run_units() {   # $1: lines file '<sbatch line> #MARK <marker>'
  local line cmd mark j left n=0 lim=$(( WAIT_H * 3600 / (POLL > 0 ? POLL : 1) )); declare -A JOB TRY CMD
  while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue; cmd=${line%% #MARK *}; mark=${line##* #MARK }; CMD[$mark]=$cmd; TRY[$mark]=0
    if [ -f "$mark" ]; then say "already complete: $mark"; JOB[$mark]=""; continue; fi
    subline "$cmd"; JOB[$mark]=$ID
  done < "$1"
  while :; do
    left=0
    for mark in "${!CMD[@]}"; do
      [ -f "$mark" ] && continue; left=$((left + 1)); j=${JOB[$mark]}
      if [ -z "$j" ] || gone "$j"; then
        soon "$mark" && continue
        if [ "${TRY[$mark]}" -ge "$RETRIES" ]; then say "STOP: unit failed $((RETRIES + 1)) times without $mark (last job $j)"; exit 1; fi
        TRY[$mark]=$((TRY[$mark] + 1)); say "RETRY ${TRY[$mark]}/$RETRIES: job $j left the queue without $mark"; subline "${CMD[$mark]}"; JOB[$mark]=$ID
      fi
    done
    [ "$left" -eq 0 ] && break
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$lim" ] && { say "STOP: $left units incomplete after $WAIT_H h"; exit 1; }
  done
}
say "start pid $$ (D1 root $D1, R7-40 root $R, CAP $CAP, WAIT_H $WAIT_H h, POLL $POLL s, RETRIES $RETRIES, S4 rule $S4_RULE, S1 rule $S1_RULE, partitions probe $PROBE_PART teacher $TEACHER_PART cache $CACHE_PART selectors $SEL_PART)"
case "$STAGE" in
  40a)
    grep -q 'r7_40a_staged_freeze' "$P" || { say "STOP: R7-40a is not frozen ($P has no r7_40a_staged_freeze)"; exit 1; }
    grep -q '"frozen": true' "$D1/data/d1_manifest_meta.json" 2>/dev/null || { say "STOP: frozen D1 manifest not found"; exit 1; }
    plan probe --s4-rule "$S4_RULE" --s1-rule "$S1_RULE" --partition "$PROBE_PART" --lines "$LD/r7_d1_40a_probe_lines.txt"; run_units "$LD/r7_d1_40a_probe_lines.txt"
    grep -q '"pass": true' "$R/probe/probe_report.json" || { say "STOP: the launch probe did not pass under S4 rule $S4_RULE / S1 rule $S1_RULE ($R/probe/probe_report.json)"; exit 1; }
    say "stage 0 complete: launch probe passes"
    plan teacher --manifest-dir "$D1/data" --partition "$TEACHER_PART" --lines "$LD/r7_d1_40a_teacher_lines.txt"; run_units "$LD/r7_d1_40a_teacher_lines.txt"
    say "stage 1 complete: teacher bundles"
    subwait --job-name=r7d40_complete $CPUJ slurm/r7_d1_40_cpu.sbatch complete
    soon "$R/teacher/r7_40a_completeness.json" && grep -q '"complete": true' "$R/teacher/r7_40a_completeness.json" \
      || { say "STOP: completeness check failed ($R/teacher/r7_40a_completeness.json)"; exit 1; }
    subwait --job-name=r7d40_readout $CPUJ slurm/r7_d1_40_cpu.sbatch readout
    soon "$R/teacher/r7_40a_results.json" || { say "STOP: readout job $ID ended without r7_40a_results.json"; exit 1; }
    say "done: R7-40a complete; CAL readout in $R/teacher/r7_40a_results.{json,md} (R7-40b needs its own staged freeze)";;
  40b)
    grep -q 'r7_40b_staged_freeze' "$P" || { say "STOP: R7-40b is not frozen ($P has no r7_40b_staged_freeze)"; exit 1; }
    grep -q '"complete": true' "$R/teacher/r7_40a_completeness.json" 2>/dev/null || { say "STOP: R7-40a is not complete"; exit 1; }
    plan cache --partition "$CACHE_PART" --lines "$LD/r7_d1_40b_cache_lines.txt"; run_units "$LD/r7_d1_40b_cache_lines.txt"; say "stage 0 complete: feature cache"
    plan stage1 --partition "$SEL_PART" --lines "$LD/r7_d1_40b_stage1_lines.txt"; run_units "$LD/r7_d1_40b_stage1_lines.txt"; say "stage 1 complete: 6 seed-0 units"
    subwait --job-name=r7d40_choose_lr $CPUJ slurm/r7_d1_40_cpu.sbatch choose
    soon "$R/selectors/lr_choice.json" || { say "STOP: choice job $ID ended without lr_choice.json"; exit 1; }
    say "lr choice: $(tr -d '\n ' < "$R/selectors/lr_choice.json" | cut -c1-300)"
    plan stage2 --selectors-dir "$R/selectors" --partition "$SEL_PART" --lines "$LD/r7_d1_40b_stage2_lines.txt"; run_units "$LD/r7_d1_40b_stage2_lines.txt"
    say "done: R7-40b complete (12 units)";;
  *) say "STOP: usage r7_d1_40_submit.sh 40a|40b"; exit 2;;
esac
