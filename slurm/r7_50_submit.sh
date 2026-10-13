#!/usr/bin/env bash
# R7-50 one-shot official-test driver (login node), one PHASE per run: L (default) | F | Q.  Run ONLY after the coordinator has frozen and
# committed the R7-50 protocol and code:
#   cd /home/infres/yinwang/CS_QMI/ssl_pilot && nohup setsid bash slurm/r7_50_submit.sh >/dev/null 2>&1 &                       (phase L)
#   PHASE=F F_LR_CHOICE=<D1-FULL lr_choice.json> F_SELECTORS=<dir> F_PCA=<pca.pt> F_INTEGRITY_REF=<D1-FULL D1-30-style DEV / val pred dir>
#       F_TIMING_DIR=<D1-FULL timing run, D1-30 layout> nohup setsid bash slurm/r7_50_submit.sh >/dev/null 2>&1 &                      (phase F)
#   stop by PID (kill <pid>, never pkill -f); log slurm_logs/r7_50_submit_<PHASE>.log (its first line carries the PID)
# Gate 0 (every phase): the protocol ($PROTOCOL) carries 'status: frozen' and a top-level 'r7_50_freeze_<yyyymmdd>:' key, and it, the R7-50
#   code, the imported D1 / D1-30 modules and R7_DATA_CLOSED_SHA256.csv are committed and unchanged (git); else STOP before anything is read.
# Phase L  gate 1: D1-20 complete (the D1-30 driver's file gates: lr_choice.json, the 15 units' .pt / .json with CAL dynamic taus) AND
#            'stage 1 complete' in slurm_logs/r7_d1_30_submit.log AND D1-30's timing files (gqa_dev, chartqa_val).
#          integrity gate (owner directive 2026-10-12): the R7-50 path on the FULL GQA DEV and ChartQA val sets with D1-30's ids, reproducing
#            D1-30's sharding / chunking on the GPU class of each reference shard; full, grid-144, SparseVLM-192 and S1 seed 0 at 25 % must equal
#            D1-30's answers item by item; any mismatch -> STOP, before any test file is opened.
#          stage 0: prepare = build (the ONLY job that opens the two test files) + configs_L + lines_L.
# Phase F  gate 1: D1-FULL complete (F_LR_CHOICE, its 9 units' .pt / .json with CAL dynamic taus in F_SELECTORS, F_PCA), D1-FULL's timing run
#            (F_TIMING_DIR/<gqa_dev|chartqa_val>/timing_<bench>.json, coordinator decision F3) AND phase L's sealed inputs; integrity gate on
#            F_INTEGRITY_REF (D1-FULL's own D1-30-style full-scale DEV / val outputs, S1 seed 0 at 25 %; 'none' is not allowed, decision F2);
#            stage 0: configs_F + lines_F (no test file is opened; inputs are phase L's).
# Phase Q  STOP: needs its protocol addendum and the R7-40c Qwen runner.
# Then (L, F): generation units below CAP with completion markers (resubmitted at most RETRIES times, then STOP) -> manifest_<P> (CPU) ->
#   STOP AND WAIT for $ROOT/ALLOW_SCORING_<P> (the coordinator commits a copy of the manifest, then writes manifest_sha256=, commit= and
#   manifest_copy= into it; the scorer checks the committed bytes) -> score (CPU; the only job that opens the sealed labels).
# No afterok chains; an empty / non-numeric job id stops the driver; submissions only while fewer than CAP (26) normal-QOS jobs are queued.
# Dry-run overrides (stubbed sbatch / squeue): R750_DRYRUN=1 (skips the git check, refuses the production root), R750_ROOT, D1_ROOT, D30_LOG,
# SUBMIT_LOG, CAP, WAIT_H, WAIT_SCORE_H, POLL, PY, PROTOCOL, RETRIES.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
PHASE=${PHASE:-L}; PROD_ROOT=/home/infres/yinwang/CS_QMI/outputs/R7_50
ROOT=${R750_ROOT:-$PROD_ROOT}; CAP=${CAP:-26}; WAIT_H=${WAIT_H:-96}; WAIT_SCORE_H=${WAIT_SCORE_H:-336}; POLL=${POLL:-300}; RETRIES=${RETRIES:-2}
PY=${PY:-/home/infres/yinwang/CS_QMI/env/bin/python}; D1=${D1_ROOT:-/home/infres/yinwang/CS_QMI/outputs/R7_D1}; D30=$D1/d1_30
P=${PROTOCOL:-reports/R7/R7_50_PROTOCOL_FROZEN_20261012.yaml}
D30LOG=${D30_LOG:-/home/infres/yinwang/CS_QMI/slurm_logs/r7_d1_30_submit.log}
LOG=${SUBMIT_LOG:-/home/infres/yinwang/CS_QMI/slurm_logs/r7_50_submit_${PHASE}.log}
CPUJ="--partition=CPU --gres=gpu:0 --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
lim() { echo $(( $1 * 3600 / (POLL > 0 ? POLL : 1) )); }
waitcap() { while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -ge "$CAP" ]; do sleep "$POLL"; done; }
subwait() { waitcap; local out rc; out=$(sbatch --parsable --wait "$@"); rc=$?; ID=$(echo "$out" | head -1); ID=${ID%%;*}   # blocks until the job ends
  [[ "$ID" =~ ^[0-9]+$ ]] || { say "STOP: no job id from sbatch --wait $*"; exit 1; }; say "$ID $* (exit $rc)"
  [ "$rc" -eq 0 ] || { say "STOP: job $ID exited $rc"; exit 1; }; }
subline() { waitcap; ID=$(eval "$1") || ID=""; [[ "$ID" =~ ^[0-9]+$ ]] || { say "STOP: no job id from: $1"; exit 1; }; say "$ID $1"; }
gone() { local out rc; out=$(squeue -h -j "$1" -o %T 2>&1); rc=$?          # left the queue: listed empty, or purged ("Invalid job id");
  if [ $rc -eq 0 ]; then [ -z "$out" ]; return; fi; [[ "$out" == *"Invalid job id"* ]]; }   # any other squeue error counts as still queued
gate() { local n=0 l; l=$(lim "$WAIT_H"); until [ -f "$1" ]; do
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$l" ] && { say "STOP: $1 missing after $WAIT_H h"; exit 1; }; done; }
gatedyn() { local n=0 l; l=$(lim "$WAIT_H"); until grep -q '"dynamic"' "$1" 2>/dev/null; do   # unit log rewritten with the CAL taus
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$l" ] && { say "STOP: no dynamic taus in $1 after $WAIT_H h"; exit 1; }; done; }
gatelog() { local n=0 l; l=$(lim "$WAIT_H"); until grep -q "$2" "$1" 2>/dev/null; do
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$l" ] && { say "STOP: '$2' not in $1 after $WAIT_H h"; exit 1; }; done; }
run_lines() {   # $1 lines file, $2 label: every unit without its marker is submitted; one whose job left the queue without it is resubmitted
  local file=$1 label=$2 line cmd mark n=0 l left j; declare -A JOB=() TRY=() CMD=()
  while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue; cmd=${line%% #MARK *}; mark=${line##* #MARK }
    CMD[$mark]=$cmd; TRY[$mark]=0
    if [ -f "$mark" ]; then say "$label: already complete: $mark"; JOB[$mark]=""; continue; fi
    subline "$cmd"; JOB[$mark]=$ID
  done < "$file"
  l=$(lim "$WAIT_H")
  while :; do
    left=0
    for mark in "${!CMD[@]}"; do
      [ -f "$mark" ] && continue; left=$((left + 1)); j=${JOB[$mark]}
      if [ -z "$j" ] || gone "$j"; then
        if [ "${TRY[$mark]}" -ge "$RETRIES" ]; then say "STOP: $label unit failed $((RETRIES + 1)) times without $mark (last job $j)"; exit 1; fi
        TRY[$mark]=$((TRY[$mark] + 1)); say "RETRY ${TRY[$mark]}/$RETRIES: job $j left the queue without $mark"; subline "${CMD[$mark]}"; JOB[$mark]=$ID
      fi
    done
    [ "$left" -eq 0 ] && break
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$l" ] && { say "STOP: $left $label units incomplete after $WAIT_H h"; exit 1; }
  done
  say "$label complete (${#CMD[@]} units)"; }
say "start pid $$ (phase $PHASE, root $ROOT, protocol $P, CAP $CAP, WAIT_H $WAIT_H h, WAIT_SCORE_H $WAIT_SCORE_H h, POLL $POLL s, RETRIES $RETRIES)"
case "$PHASE" in L|F) ;; Q) say "STOP: phase Q (Qwen2.5-VL) needs its protocol addendum and the R7-40c runner; rows are declared in the protocol"; exit 1;;
  *) say "STOP: unknown PHASE $PHASE"; exit 1;; esac
# gate 0 — frozen protocol, committed code
[ -f "$P" ] && grep -qE '^status:[[:space:]]*frozen' "$P" && grep -qE '^r7_50_freeze_[0-9]{8}[[:space:]]*:' "$P" \
  || { say "STOP: $P is not a frozen R7-50 protocol (status: frozen + top-level r7_50_freeze_<yyyymmdd>: key); nothing read"; exit 1; }
if [ "${R750_DRYRUN:-0}" = 1 ]; then
  [ "$ROOT" != "$PROD_ROOT" ] || { say "STOP: a dry run refuses the production root"; exit 1; }
  say "DRY RUN: git check skipped"
else
  FILES="$P scripts/r7_50_test.py slurm/r7_50_cpu.sbatch slurm/r7_50_gpu.sbatch slurm/r7_50_submit.sh scripts/r7_d1_30_data.py scripts/r7_d1_30_plan.py
         scripts/r7_d1_30_run.py scripts/r7_d1_30_score.py scripts/r7_d1_30_sparsevlm.py scripts/r7_d1_30_eval.py scripts/r7_d1_data.py scripts/r7_d1_masks.py
         scripts/r7_d1_selectors.py scripts/r7_d1_teacher.py scripts/r7_vl_common.py reports/R7/R7_DATA_CLOSED_SHA256.csv"
  git ls-files --error-unmatch $FILES >/dev/null 2>&1 || { say "STOP: protocol / code not committed"; exit 1; }
  git diff --quiet HEAD -- $FILES || { say "STOP: protocol / code differ from HEAD"; exit 1; }
  say "gate 0 ok: git HEAD $(git rev-parse HEAD)"
fi
IARGS=()
if [ "$PHASE" = L ]; then
  # gate 1 — D1-20 complete (the D1-30 driver's file gates) and D1-30 stage 1 complete
  gate "$D1/selectors/lr_choice.json"
  UNITS=$("$PY" -I scripts/r7_d1_30_plan.py units --lr-choice "$D1/selectors/lr_choice.json") || { say "STOP: unit list"; exit 1; }
  for u in $UNITS; do gate "$D1/selectors/$u.pt"; gatedyn "$D1/selectors/$u.json"; done
  say "D1-20 complete: $(echo $UNITS | wc -w) units"
  # execution note r7_50_1 (coordinator 2026-10-12): wait for exactly the D1-30 outputs R7-50 relies on — the GQA DEV / ChartQA val quality and
  # SparseVLM shards (the integrity gate's references) — instead of all of D1-30; the timing files move to just before scoring
  for i in 0 1 2 3 4 5; do gate "$D30/pred/gqa_dev/done_shard${i}of6.json"; done
  for i in 0 1; do gate "$D30/pred/chartqa_val/done_shard${i}of2.json"; done
  for i in 0 1 2; do gate "$D30/pred/gqa_dev/sparsevlm_run_gqa_dev_shard${i}of3.json"; done
  gate "$D30/pred/chartqa_val/sparsevlm_run_chartqa_val_shard0of1.json"
  say "D1-30 gqa_dev / chartqa_val quality and SparseVLM shards complete (execution note r7_50_1); timing is gated before scoring"
else
  for v in F_LR_CHOICE F_SELECTORS F_PCA F_INTEGRITY_REF F_TIMING_DIR; do [ -n "${!v}" ] || { say "STOP: phase F needs $v"; exit 1; }; done
  [ "$F_INTEGRITY_REF" != none ] || { say "STOP: F_INTEGRITY_REF=none is not allowed (coordinator decision F2)"; exit 1; }
  for b in gqa_dev chartqa_val; do gate "$F_TIMING_DIR/$b/timing_$b.json"; done
  gate "$F_LR_CHOICE"; gate "$F_PCA"
  UNITS=$("$PY" -I scripts/r7_50_test.py units --lr-choice "$F_LR_CHOICE") || { say "STOP: D1-FULL unit list"; exit 1; }
  for u in $UNITS; do gate "$F_SELECTORS/$u.pt"; gatedyn "$F_SELECTORS/$u.json"; done
  grep -q '"ok": true' "$ROOT/sealed/inputs/build_summary.json" 2>/dev/null || { say "STOP: phase F needs phase L's sealed inputs ($ROOT/sealed/inputs)"; exit 1; }
  say "D1-FULL complete: $(echo $UNITS | wc -w) units; sealed inputs present"
  IARGS=("$F_LR_CHOICE" "$F_SELECTORS" "$F_PCA" "$F_INTEGRITY_REF")
fi
# full-scale integrity gate (non-test data; before any test file is opened in phase L, before any generation in phase F)
I=$ROOT/integrity_$PHASE
if "$PY" -I -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("pass") is True else 1)' "$I/integrity.json" 2>/dev/null; then   # top-level key only (note r7_50_7)
  say "integrity gate already passed ($I/integrity.json)"
else
  [ -f "$I/lines.txt" ] || subwait --job-name=r7_50_iprep_$PHASE $CPUJ slurm/r7_50_cpu.sbatch integrity-prepare "$ROOT" "$PHASE" "${IARGS[@]}"
  [ -f "$I/lines.txt" ] || { say "STOP: $I/lines.txt missing"; exit 1; }
  run_lines "$I/lines.txt" "integrity"
  waitcap; out=$(sbatch --parsable --wait --job-name=r7_50_icmp_$PHASE $CPUJ slurm/r7_50_cpu.sbatch integrity-compare "$ROOT" "$PHASE"); rc=$?
  for i in $(seq 1 24); do [ -f "$I/integrity.json" ] && break; sleep 5; done   # NFS (note r7_50_7)
  "$PY" -I -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("pass") is True else 1)' "$I/integrity.json" 2>/dev/null || { say "STOP: full-scale integrity gate FAILED (job ${out%%;*}, exit $rc; see $I/integrity.json); no test file opened, nothing generated"; exit 1; }
  say "integrity gate passed (job ${out%%;*}; $I/integrity.json)"
fi
# stage 0 — inputs (phase L: the one read of each test file) and the phase's configs / lines
if [ "$PHASE" = L ] && [ ! -f "$ROOT/sealed/inputs/build_summary.json" ]; then
  subwait --job-name=r7_50_prepare $CPUJ slurm/r7_50_cpu.sbatch prepare "$ROOT" "$P"
elif [ ! -f "$ROOT/lines_$PHASE.txt" ]; then
  if [ "$PHASE" = L ]; then subwait --job-name=r7_50_configs_L $CPUJ slurm/r7_50_cpu.sbatch configs "$ROOT" L
  else subwait --job-name=r7_50_configs_F $CPUJ slurm/r7_50_cpu.sbatch configs "$ROOT" F "$F_LR_CHOICE" "$F_SELECTORS" "$F_PCA"; fi
fi
grep -q '"ok": true' "$ROOT/sealed/inputs/build_summary.json" 2>/dev/null || { say "STOP: no ok build summary"; exit 1; }
[ -f "$ROOT/lines_$PHASE.txt" ] || { say "STOP: $ROOT/lines_$PHASE.txt missing"; exit 1; }
say "stage 0 complete: sealed inputs, configs_$PHASE, $(grep -vc '^#' "$ROOT/lines_$PHASE.txt") unit lines"
# stage 1 — generation
run_lines "$ROOT/lines_$PHASE.txt" "generation $PHASE"
# stage 2 — manifest (CPU, waits)
M=$ROOT/manifest_$PHASE
XT=""; [ "$PHASE" = F ] && XT=$F_TIMING_DIR
[ -f "$M/predictions_manifest.sha256" ] || subwait --job-name=r7_50_manifest_$PHASE $CPUJ slurm/r7_50_cpu.sbatch manifest "$ROOT" "$PHASE" $XT
[ -f "$M/predictions_manifest.json" ] && [ -f "$M/predictions_manifest.sha256" ] || { say "STOP: manifest_$PHASE missing"; exit 1; }
MSHA=$(cut -d' ' -f1 "$M/predictions_manifest.sha256")
say "stage 2 complete: manifest_$PHASE sha256 $MSHA.  WAITING for $ROOT/ALLOW_SCORING_$PHASE (coordinator: commit a copy of the manifest, then write manifest_sha256=$MSHA, commit=<that commit> and manifest_copy=<repo path of the copy> into it)"
# stage 3 — wait for the coordinator's gate file
A=$ROOT/ALLOW_SCORING_$PHASE; n=0; l=$(lim "$WAIT_SCORE_H")
until [ -f "$A" ]; do sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$l" ] && { say "STOP: no $A after $WAIT_SCORE_H h"; exit 1; }; done
grep -q "^manifest_sha256=$MSHA\$" "$A" || { say "STOP: $A does not carry manifest_sha256=$MSHA; labels stay sealed"; exit 1; }
[ "$PHASE" = L ] || [ -f "$ROOT/manifest_L/predictions_manifest.json" ] || { say "STOP: phase F scoring needs phase L's sealed baseline rows (manifest_L)"; exit 1; }
say "stage 3: $(basename "$A") present ($(grep -v '^manifest_sha256=' "$A" | tr '\n' ' ' | cut -c1-200))"
# stage 4 — score (CPU, waits): the only job that opens the sealed labels
if [ "$PHASE" = L ]; then   # execution note r7_50_1: the profiling-workload timing is needed only by the scorer
  for b in gqa_dev chartqa_val; do gate "$D30/timing/$b/timing_$b.json"; gate "$D30/timing/$b/sparsevlm_timing_$b.json"; done
  say "profiling-workload timing present (D1-30 gqa_dev / chartqa_val)"
fi
subwait --job-name=r7_50_score_$PHASE $CPUJ slurm/r7_50_cpu.sbatch score "$ROOT" "$PHASE" $XT
[ -f "$ROOT/results_$PHASE/r7_50_results.json" ] || { say "STOP: score job $ID ended without results_$PHASE/r7_50_results.json"; exit 1; }
say "done: $ROOT/results_$PHASE/r7_50_results.json and .md"
