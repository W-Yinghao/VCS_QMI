#!/usr/bin/env bash
# Resume ONE byte-range part until it is complete (the CDN closes streams after 1-3 GB with rc 0).  Usage: fetch_part_loop.sh <i>
URL=https://ncr-paper.cdn.bcebos.com/data/NCR-data.tar; TOTAL=100154890240; N=8; i=$1; D=$(dirname "$0"); LOG=/home/infres/yinwang/CS_QMI/slurm_logs/v9_fetch_ncr.log
seg=$(( (TOTAL + N - 1) / N )); a=$((i*seg)); b=$(( (i+1)*seg - 1 )); [ $b -ge $TOTAL ] && b=$((TOTAL-1)); need=$((b-a+1)); f=$D/part_$i
for try in $(seq 1 200); do
  have=0; [ -f $f ] && have=$(stat -c %s $f); [ $have -ge $need ] && { echo "$(date -u +%FT%TZ) part $i complete ($have)" >> $LOG; exit 0; }
  curl -s -L --retry 5 --retry-delay 5 -r $((a+have))-$b -o - "$URL" >> $f; echo "$(date -u +%FT%TZ) part $i try $try rc=$? size=$(stat -c %s $f)/$need" >> $LOG; sleep 3
done; exit 1
