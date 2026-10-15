#!/usr/bin/env bash
# Waits for the first fetch instance, then runs per-part resume loops for every incomplete part (one writer per part), then concatenates.
D=$(dirname "$0"); LOG=/home/infres/yinwang/CS_QMI/slurm_logs/v9_fetch_ncr.log; FIRST=$1; TOTAL=100154890240; N=8; seg=$(( (TOTAL + N - 1) / N ))
while kill -0 "$FIRST" 2>/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) supervise2: first instance gone; launching per-part loops" >> $LOG
pids=(); for i in $(seq 0 $((N-1))); do pgrep -f "fetch_part_loop.sh $i\$" > /dev/null && continue; bash $D/fetch_part_loop.sh $i & pids+=($!); done
wait "${pids[@]}"; while pgrep -f "fetch_part_loop.sh" > /dev/null; do sleep 30; done
ok=1; for i in $(seq 0 $((N-1))); do a=$((i*seg)); b=$(( (i+1)*seg - 1 )); [ $b -ge $TOTAL ] && b=$((TOTAL-1)); [ "$(stat -c %s $D/part_$i)" -eq $((b-a+1)) ] || ok=0; done
if [ $ok -eq 1 ] && [ ! -f $D/../NCR-data.tar ]; then cat $D/part_* > $D/../NCR-data.tar && [ "$(stat -c %s $D/../NCR-data.tar)" -eq $TOTAL ] && rm -f $D/part_* && echo "$(date -u +%FT%TZ) CONCAT OK $(stat -c %s $D/../NCR-data.tar)" >> $LOG; else echo "$(date -u +%FT%TZ) supervise2: ok=$ok tar_exists=$([ -f $D/../NCR-data.tar ] && echo 1 || echo 0)" >> $LOG; fi
