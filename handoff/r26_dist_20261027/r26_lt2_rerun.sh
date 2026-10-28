#!/usr/bin/env bash
# Watches the two R26 runs that started before the skip_labeled_lt2 fix (d_vcs, d_bce at f40_s1001). If one leaves the
# queue without DONE.json AND its EPOCHS.json shows a 1-sample labeled-batch event (split % 128 == 1), the run dir is
# archived under attempts/ and the same line is resubmitted from the same prefix (fixed code). Any other failure is only
# logged (no blind rerun). Exits when both runs are DONE or handled. STOP_LT2 stops it. Log: lt2_rerun.log.
D=/home/infres/yinwang/CS_QMI/handoff/r26_dist_20261027
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r26_dist_20261027/runs/f40_s1001
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
log() { echo "$(date -u +%FT%TZ) $*" >> $D/lt2_rerun.log; }
log "start pid $$"
declare -A NAME=([d_vcs]=r26_dvcs_f40_s1001 [d_bce]=r26_dbce_f40_s1001)
while [ ! -e $D/STOP_LT2 ]; do
  left=0
  for arm in d_vcs d_bce; do
    [ -f $D/LT2_HANDLED_$arm ] && continue
    [ -f $O/$arm/DONE.json ] && { log "$arm DONE without rerun"; touch $D/LT2_HANDLED_$arm; continue; }
    left=1
    squeue -u $USER -h -o %j | grep -qx ${NAME[$arm]} && continue
    ev=$($PY -c "
import json;co=json.load(open('$O/$arm/EPOCHS.json'))['cotrain']
print(sum(1 for e in co for sp in ('split_A','split_B') if e[sp]['n_pred_clean']%128==1))" 2>/dev/null || echo err)
    if [ "$ev" != "0" ] && [ "$ev" != "err" ]; then
      J0=$(ls $O/$arm/RUN_*.json | tail -1 | grep -oE '[0-9]+'); mkdir -p $O/attempts
      mv $O/$arm $O/attempts/${arm}_${J0}_labeled1_nan
      L=$(grep -- "--job-name=${NAME[$arm]} " $D/LINES_A.txt)
      J=$(cd /home/infres/yinwang/CS_QMI/slurm_logs && eval "${L/sbatch /sbatch --parsable }")
      printf "%s\t%s\t%s\trerun from same prefix with fixed code after 1-sample labeled-batch event (attempt archived %s)\n" "$(date -u +%FT%TZ)" ${NAME[$arm]} "$J" "attempts/${arm}_${J0}_labeled1_nan" >> $D/SUBMITTED.tsv
      log "$arm left queue with $ev event(s): archived, resubmitted $J"; touch $D/LT2_HANDLED_$arm
    else
      log "$arm left queue without DONE and without a 1-sample event (events=$ev): not rerun, needs inspection"; touch $D/LT2_HANDLED_$arm
    fi
  done
  [ $left -eq 0 ] && { log "both handled: exit"; exit 0; }
  sleep 300
done
log "STOP_LT2"
