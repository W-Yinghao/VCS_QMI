#!/usr/bin/env bash
# Re-admits verified nodes to my PENDING jobs without touching any frozen driver: every 60 s, each pending normal-QOS job of mine whose
# ExcNodeList contains one of $ALLOW (comma list, e.g. node52) gets that node removed (the other exclusions stay).  Ordering / placement only.
# Stops after 24 h without a job to update.  Stop by PID.  Log: slurm_logs/exc_updater.log
ALLOW=${ALLOW:?set ALLOW, e.g. node52}; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/exc_updater.log; idle=0
echo "$(date -u +%FT%TZ) start pid $$ allow $ALLOW" >> $LOG
while [ $idle -lt 1440 ]; do
  upd=0
  for j in $(squeue -h -u yinwang -q normal -t PD -o "%i"); do
    exc=$(scontrol show job "$j" 2>/dev/null | grep -oE "ExcNodeList=[^ ]+" | cut -d= -f2); [ -z "$exc" ] || [ "$exc" = "(null)" ] && continue
    full=$(scontrol show hostnames "$exc" | paste -sd, -)                     # node[51-52,60] -> node51,node52,node60
    new=$full; for n in ${ALLOW//,/ }; do new=$(echo ",$new," | sed "s/,$n,/,/g; s/^,//; s/,$//"); done
    [ "$new" = "$full" ] && continue
    if true; then
      [ -z "$new" ] && { echo "$(date -u +%FT%TZ) skip $j (would clear every exclusion)" >> $LOG; continue; }
      scontrol update jobid="$j" ExcNodeList="$new" && echo "$(date -u +%FT%TZ) $j ExcNodeList $full -> $new" >> $LOG; upd=1
    fi
  done
  [ $upd = 0 ] && idle=$((idle + 1)) || idle=0; sleep 60
done; echo "$(date -u +%FT%TZ) done" >> $LOG
