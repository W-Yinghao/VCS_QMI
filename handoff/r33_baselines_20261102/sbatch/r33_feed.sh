#!/bin/bash
# R33 feeder (no --dependency): every 120 s, in CELLS.json order, submit the cell prefix if missing, then each arm of a cell whose
# prefix is done, when no job of that name is queued/running and my runfill job count < CAP. Unfinished units without a live job
# (timeout continuation failed, preempted-and-cancelled, ...) are resubmitted and resume from their own state. A unit/prefix with
# >= 3 lines in FAILED.tsv is left alone (logged as BLOCKED). Exits when every prefix/unit is done or blocked.
# STOP_FEED -> exit; PAUSE_FEED -> no submissions.
D=/home/infres/yinwang/CS_QMI/handoff/r33_baselines_20261102
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r33_baselines_20261102
CAP=${CAP:-27}
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
LIST=$($PY -B -c "
import json; c=json.load(open('$D/CELLS.json'))
for k in c['_order']: print(k, ' '.join(c[k]['arms']))")
log() { echo "$(date -u +%FT%TZ) $*" >> $D/feed.log; }
log "feeder start job ${SLURM_JOB_ID:-none} cap $CAP"
while true; do
  [ -f $D/STOP_FEED ] && { log "STOP_FEED"; exit 0; }
  Q=$(squeue -u yinwang -h -o "%j|%q" 2>/dev/null) || { log "squeue failed; skip"; sleep 120; continue; }
  [ -z "$Q" ] && { log "squeue empty output; skip"; sleep 120; continue; }   # never infer an empty queue
  NRF=$(echo "$Q" | grep -c '|runfill$')
  pending=0; live=0; blocked=""
  while read -r CELL ARMS; do
    PD=$O/prefix/$CELL
    if [ -f $PD/PREFIX_DONE.json ]; then KINDS="$ARMS"; else KINDS=prefix; fi
    for K in $KINDS; do
      if [ $K = prefix ]; then OUT=$PD; else OUT=$O/runs/$CELL/$K; [ -f $OUT/DONE.json ] && continue; fi
      pending=1
      NAME=r33_${K}_${CELL}
      echo "$Q" | grep -q "^$NAME|" && { live=1; continue; }
      NF=0; [ -f $OUT/FAILED.tsv ] && NF=$(wc -l < $OUT/FAILED.tsv)
      [ $NF -ge 3 ] && { blocked="$blocked $NAME"; continue; }
      live=1
      [ -f $D/PAUSE_FEED ] && continue
      [ $NRF -ge $CAP ] && continue
      J=$(sbatch --parsable --job-name=$NAME $D/sbatch/r33_unit.sbatch $K $CELL 2>>$D/feed.log </dev/null) || { log "sbatch failed $NAME"; continue; }
      NRF=$((NRF+1)); printf "%s\t%s\t%s\t%s\t%s\tfeeder\n" "$(date -u +%FT%TZ)" "$NAME" "$J" "$K" "$CELL" >> $D/SUBMITTED.tsv
      log "submitted $NAME $J (runfill now $NRF)"
    done
  done <<< "$LIST"
  [ -n "$blocked" ] && log "BLOCKED:$blocked"
  if [ $pending = 0 ]; then log "all prefixes/units done"; exit 0; fi
  if [ $live = 0 ] && [ -n "$blocked" ]; then log "only blocked items left; exit"; exit 1; fi
  sleep 120
done
