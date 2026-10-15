#!/usr/bin/env bash
# Re-run fetch_parallel.sh (resumable per part) until the concatenation succeeds; waits for the running instance first.
D=$(dirname "$0"); LOG=/home/infres/yinwang/CS_QMI/slurm_logs/v9_fetch_ncr.log; FIRST=${1:?pid of the running instance}
while kill -0 "$FIRST" 2>/dev/null; do sleep 30; done
for try in $(seq 1 30); do
  grep -q "CONCAT OK" $LOG && { echo "$(date -u +%FT%TZ) supervisor: done" >> $LOG; exit 0; }
  echo "$(date -u +%FT%TZ) supervisor: rerun $try" >> $LOG; bash $D/fetch_parallel.sh; sleep 10
done
echo "$(date -u +%FT%TZ) supervisor: gave up after 30 reruns" >> $LOG; exit 1
