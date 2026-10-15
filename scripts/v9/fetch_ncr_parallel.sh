#!/usr/bin/env bash
# V9: parallel byte-range download of NCR-data.tar (100,154,890,240 bytes; CDN allows ranges; ~5 MB/s per stream) into 8 part files, then
# concatenated into NCR-data.tar for data_fetch.py (sha256 + safe extraction).  Resumable per part (curl -C -).
URL=https://ncr-paper.cdn.bcebos.com/data/NCR-data.tar; TOTAL=100154890240; N=8; D=$(dirname "$0"); LOG=/home/infres/yinwang/CS_QMI/slurm_logs/v9_fetch_ncr.log
echo "$(date -u +%FT%TZ) start N=$N total=$TOTAL pid=$$" >> $LOG
seg=$(( (TOTAL + N - 1) / N )); pids=()
for i in $(seq 0 $((N-1))); do a=$((i*seg)); b=$(( (i+1)*seg - 1 )); [ $b -ge $TOTAL ] && b=$((TOTAL-1)); f=$D/part_$i; have=0; [ -f $f ] && have=$(stat -c %s $f)
  need=$((b-a+1)); if [ $have -ge $need ]; then echo "$(date -u +%FT%TZ) part $i complete" >> $LOG; continue; fi
  ( curl -s -L --retry 20 --retry-delay 10 -r $((a+have))-$b -o - "$URL" >> $f; echo "$(date -u +%FT%TZ) part $i rc=$? size=$(stat -c %s $f) need=$need" >> $LOG ) & pids+=($!)
done
wait "${pids[@]}"
ok=1; for i in $(seq 0 $((N-1))); do a=$((i*seg)); b=$(( (i+1)*seg - 1 )); [ $b -ge $TOTAL ] && b=$((TOTAL-1)); [ "$(stat -c %s $D/part_$i)" -eq $((b-a+1)) ] || ok=0; done
if [ $ok -eq 1 ]; then cat $D/part_* > $D/../NCR-data.tar && [ "$(stat -c %s $D/../NCR-data.tar)" -eq $TOTAL ] && rm -f $D/part_* && echo "$(date -u +%FT%TZ) CONCAT OK $(stat -c %s $D/../NCR-data.tar)" >> $LOG; else echo "$(date -u +%FT%TZ) INCOMPLETE parts; rerun to resume" >> $LOG; fi
