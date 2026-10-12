#!/usr/bin/env bash
# Execution note E3 recovery (2026-10-12, coordinator): the 7 early SparseVLM shards ran pre-E3 code and stop on the old exact counter check at
# the 432-adapted row (float32 drift of SparseVLM's own counter) after writing every answer of the measured rows.  For each shard, once its job
# has left the queue and if it did not exit 0: (1) backfill its printed row records into the E3 rows file (CPU job, waits; single writer since
# the shard job has ended); (2) resubmit the driver's own line for the shard (GQA / ChartQA: re-check + marker; TextVQA: the 4 OCR rows + marker).
# Stop by PID.  Log: slurm_logs/r7_d1_30_e3_recover.log.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1; CAP=${CAP:-27}; L=/home/infres/yinwang/CS_QMI/slurm_logs; LOG=$L/r7_d1_30_e3_recover.log
PRED=/home/infres/yinwang/CS_QMI/outputs/R7_D1/d1_30/pred; PY=/home/infres/yinwang/CS_QMI/env/bin/python
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }; say "start pid $$"
waitcap() { while [ "$(squeue -h -u "$USER" -q normal | wc -l)" -ge "$CAP" ]; do sleep 60; done; }
recover() {   # $1 job id  $2 bench  $3 shard (i/n)
  local j=$1 b=$2 s=$3 log; while squeue -h -j "$j" 2>/dev/null | grep -q .; do sleep 60; done
  log=$(ls $L/r7d30v_${b}_${s%%/*}_$j.out 2>/dev/null); [ -n "$log" ] || { say "STOP: no log for $j"; return 1; }
  if grep -q "^exit=0" "$log"; then say "$j $b $s exited 0: nothing to recover"; return 0; fi
  waitcap; local bj; bj=$(sbatch --parsable --wait --job-name=r7d30_backfill_${b}_${s%%/*} --partition=CPU --gres=gpu:0 --cpus-per-task=2 --mem=8G --time=00:20:00 \
      --output=$L/%x_%j.out --wrap "cd /home/infres/yinwang/CS_QMI && $PY -I ssl_pilot/scripts/r7_d1_30_backfill_rows.py --repo ssl_pilot --pred-root $PRED --log $log; echo exit=\$?") || bj=""
  [[ "$bj" =~ ^[0-9]+$ ]] && grep -q "^exit=0" $L/r7d30_backfill_${b}_${s%%/*}_$bj.out || { say "STOP: backfill failed for $j ($bj)"; return 1; }
  waitcap; local rj; rj=$(sbatch --parsable --job-name=r7d30v_${b}_${s%%/*}_e3 --partition=H100,L40S --exclude=node51,node52,node60 slurm/r7_d1_30_gpu.sbatch sparsevlm $b $s) || rj=""
  [[ "$rj" =~ ^[0-9]+$ ]] || { say "STOP: resubmission failed for $b $s"; return 1; }; say "$j $b $s: backfill $bj ok; resubmitted $rj"
}
recover 1035245 gqa_dev 0/3; recover 1035248 textvqa_val 0/3
recover 1035246 gqa_dev 1/3; recover 1035247 gqa_dev 2/3; recover 1035251 chartqa_val 0/1; recover 1035249 textvqa_val 1/3; recover 1035250 textvqa_val 2/3
say "done"
