#!/usr/bin/env bash
# R7-30 D1-20 driver (login node, after the coordinator's staged freeze):
#   nohup setsid bash slurm/r7_d1_20_submit.sh >/dev/null 2>&1 &      — stop by PID; log slurm_logs/r7_d1_20_submit.log
# Stage 0: the feature-cache job (LLaVA env, H100 / L40S); gate: every cache file present after the job has left the queue.
# Stage 1: the 6 learning-rate-search units (seed 0), lines from scripts/r7_d1_plan20.py stage1 — no afterok chains (a finished dependency is
#          purged after MinJobAge and sbatch then rejects it); gate: the 6 final <tag>.pt files (file gates; give up after WAIT_H hours).
# Choice:  scripts/r7_d1_plan20.py choose in a CPU sbatch job (sbatch --wait; never on the login node); gate: lr_choice.json.
# Stage 2: the 12 dependent units (seeds 1-2 at the chosen rates, A1 / A2 × seeds 0-2).
# Every submission waits below CAP; an empty or non-numeric job id stops the driver (logged).  Overrides for dry runs: D1_ROOT, SUBMIT_LOG,
# CAP, WAIT_H, POLL, PY, S1_LINES, S2_LINES.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1
CAP=${CAP:-27}; WAIT_H=${WAIT_H:-48}; POLL=${POLL:-300}; PY=${PY:-/home/infres/yinwang/CS_QMI/env/bin/python}
D1=${D1_ROOT:-/home/infres/yinwang/CS_QMI/outputs/R7_D1}; LOG=${SUBMIT_LOG:-/home/infres/yinwang/CS_QMI/slurm_logs/r7_d1_20_submit.log}
S1=${S1_LINES:-slurm/r7_d1_20_stage1_lines.txt}; S2=${S2_LINES:-slurm/r7_d1_20_stage2_lines.txt}
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
waitcap() { while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -ge "$CAP" ]; do sleep "$POLL"; done; }
sub() { waitcap; ID=$(sbatch --parsable "$@") || ID=""; [[ "$ID" =~ ^[0-9]+$ ]] || { say "STOP: no job id from sbatch $*"; exit 1; }; say "$ID $*"; }
sublines() { local line; while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue; waitcap; ID=$(eval "$line") || ID=""
    [[ "$ID" =~ ^[0-9]+$ ]] || { say "STOP: no job id from: $line"; exit 1; }; say "$ID $line"; done < "$1"; }
inq() { squeue -h -j "$1" 2>/dev/null | grep -q .; }
gate() { local n=0 lim=$(( WAIT_H * 3600 / (POLL > 0 ? POLL : 1) )); until [ -f "$1" ]; do
    sleep "$POLL"; n=$((n + 1)); [ "$n" -ge "$lim" ] && { say "STOP: $1 missing after $WAIT_H h"; exit 1; }; done; }
say "start (D1 root $D1, CAP $CAP, WAIT_H $WAIT_H h, POLL $POLL s)"
grep -q '"frozen": true' "$D1/data/d1_manifest_meta.json" 2>/dev/null || { say "STOP: frozen D1 manifest not found"; exit 1; }
# stage 0 — feature cache
sub --job-name=r7d1c_cache --partition=H100,L40S --exclude=node51,node52,node60 slurm/r7_d1_gpu.sbatch cache d1_20; CID=$ID
while inq "$CID"; do sleep "$POLL"; done
for f in pca.pt feats_FIT.npy feats_CAL.npy images_FIT.json images_CAL.json question_ids.jsonl embed_table.pt; do
  [ -f "$D1/cache/$f" ] || { say "STOP: cache job $CID ended without $f"; exit 1; }; done
say "cache complete ($CID)"
# stage 1 — learning-rate search, seed 0
"$PY" -I scripts/r7_d1_plan20.py stage1 --lines "$S1" >> "$LOG" 2>&1 || { say "STOP: stage-1 lines"; exit 1; }
sublines "$S1"
for o in vcs logistic direct; do for lr in 0.0001 0.0003; do gate "$D1/selectors/${o}_lr${lr}_s0.pt"; done; done
say "stage 1 complete (6 seed-0 units)"
# choice — CPU job, waits for completion
sub --wait --job-name=r7d1_choose_lr --partition=CPU --gres=gpu:0 --cpus-per-task=1 --mem=2G --time=00:15:00 \
    --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out \
    --wrap "cd /home/infres/yinwang/CS_QMI/ssl_pilot && $PY -I scripts/r7_d1_plan20.py choose --selectors-dir $D1/selectors"
[ -f "$D1/selectors/lr_choice.json" ] || { say "STOP: choice job $ID ended without lr_choice.json"; exit 1; }
say "lr choice: $(tr -d '\n ' < "$D1/selectors/lr_choice.json" | cut -c1-300)"
# stage 2 — dependent units
"$PY" -I scripts/r7_d1_plan20.py stage2 --selectors-dir "$D1/selectors" --lines "$S2" >> "$LOG" 2>&1 || { say "STOP: stage-2 lines"; exit 1; }
sublines "$S2"
say "done: stage 2 submitted (12 units)"
