#!/usr/bin/env bash
# Watches R26 runs that started before the native-bug fixes (code_revisions in RUN_LOCK.json). Args: list of "name:cell:arm".
# If a watched job leaves the queue without DONE.json and either (a) its logs show the empty-caption crash
# ("Length of all samples has to be greater than 0") or (b) its EPOCHS.json shows a 1-sample labeled-batch event
# (split % 128 == 1, the size-1 NaN), the run dir is archived under attempts/ and its original feeder line is resubmitted
# from the same prefix with the fixed code. Any other failure is only logged. STOP_NATIVE_BUG stops it. Log: native_bug_rerun.log.
D=/home/infres/yinwang/CS_QMI/handoff/r26_dist_20261027
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r26_dist_20261027/runs
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
log() { echo "$(date -u +%FT%TZ) $*" >> $D/native_bug_rerun.log; }
log "start pid $$ watching: $*"
while [ ! -e $D/STOP_NATIVE_BUG ]; do
  left=0
  for spec in "$@"; do
    IFS=: read name cell arm <<< "$spec"; dir=$O/$cell/$arm
    [ -f $D/NBUG_HANDLED_$name ] && continue
    [ -f $dir/DONE.json ] && { log "$name DONE"; touch $D/NBUG_HANDLED_$name; continue; }
    left=1
    squeue -u $USER -h -o %j | grep -qx $name && continue
    cap=$(grep -l "Length of all samples has to be greater than 0" $D/logs/${name}_*.out 2>/dev/null | head -1)
    ev=$($PY -c "
import json;co=json.load(open('$dir/EPOCHS.json'))['cotrain']
print(sum(1 for e in co for sp in ('split_A','split_B') if e[sp]['n_pred_clean']%128==1))" 2>/dev/null || echo err)
    if [ -n "$cap" ] || { [ "$ev" != "0" ] && [ "$ev" != "err" ]; }; then
      why=$([ -n "$cap" ] && echo empty_caption || echo labeled1_nan)
      J0=$(ls $dir/RUN_*.json | tail -1 | grep -oE '[0-9]+'); mkdir -p $O/$cell/attempts
      mv $dir $O/$cell/attempts/${arm}_${J0}_${why}
      L=$(cat $D/LINES_A.txt $D/LINES_B.txt $D/LINES_C.txt | grep -- "--job-name=$name ")
      J=$(cd /home/infres/yinwang/CS_QMI/slurm_logs && eval "${L/sbatch /sbatch --parsable }")
      printf "%s\t%s\t%s\trerun from same prefix with fixed code after %s (attempt archived attempts/%s)\n" "$(date -u +%FT%TZ)" $name "$J" $why "${arm}_${J0}_${why}" >> $D/SUBMITTED.tsv
      log "$name: $why -> archived, resubmitted $J"; touch $D/NBUG_HANDLED_$name
    else
      log "$name left queue without DONE and without a known native-bug signature (events=$ev): not rerun, needs inspection"; touch $D/NBUG_HANDLED_$name
    fi
  done
  [ $left -eq 0 ] && { log "all watched runs handled: exit"; exit 0; }
  sleep 300
done
log STOP_NATIVE_BUG
