#!/bin/bash
# Submit the v3 probe for every existing, not yet measured state (R29 saved states, R30 parents E0, fixed E10, fixed E40),
# grouped by (cond, seed) with the common H parent of that cond/seed as the frozen reference; <= CHUNK states per job.
# One-shot (no watcher). Skips a chunk if a job with the same name is queued. On a submit failure (e.g. per-user submit
# limit) it stops; rerun later. Do not rerun for a group whose v3 jobs are still queued/running (chunk boundaries shift).
# Usage: tools/launch_v3.sh [CHUNK=4] [ONLY=<cond>_s<seed>,...]
set -uo pipefail
D=/home/infres/yinwang/CS_QMI/handoff/r30_diag_20261101
CHUNK=${1:-4}
ONLY=${2:-}
cd $D
python3 -B tools/r30_states.py states_out_v3 COVERAGE_V3.json --jobs | while read -r COND SEED REF REFID REST; do
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",${COND}_s${SEED},"* ]]; then continue; fi
  read -ra ST <<< "$REST"
  for ((i=0; i<${#ST[@]}; i+=CHUNK)); do
    part=("${ST[@]:i:CHUNK}")
    first=${part[0]%%=*}
    NAME="r30d3_${COND}_s${SEED}_${first:0:28}"
    if squeue -u $USER -h -o %j | grep -qx "$NAME"; then echo "queued already: $NAME"; continue; fi
    J=$(sbatch --parsable --job-name=$NAME sbatch/margin_gpu_v3.sbatch $COND $SEED $REF $REFID "${part[@]}") || { echo "SUBMIT_FAILED $NAME (retry later)"; exit 1; }
    printf "%s\t%s\t%s\tGPU v3 probe %s_s%s ref %s: %s\n" "$(date -u +%FT%TZ)" "$NAME" "$J" "$COND" "$SEED" "$REFID" "$(printf '%s ' "${part[@]%%=*}")" >> SUBMITTED.tsv
    echo "submitted $J $NAME"
  done
done
