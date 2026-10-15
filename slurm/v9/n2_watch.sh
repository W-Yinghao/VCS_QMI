#!/usr/bin/env bash
# V9-N2 job watcher (login node).  usage: nohup setsid bash slurm/v9/n2_watch.sh <ids_file>... >/dev/null 2>&1 &
# Feeders counted as alive: slurm/v9/n2_feed.sh and scripts/v9/n2_release.py run.  Each ids file holds "<jobid> [label]" lines and is re-read every cycle (the feeder appends to slurm_logs/v9_n2_feed_ids.txt).  Every job's log is
# checked ~60 s after it starts (Traceback / refused / Error) and again when it ends (exit= line); summary in slurm_logs/v9_n2_watch_status.json,
# log in slurm_logs/v9_n2_watch.log.  Exits when every listed job has ended and no n2_feed.sh is alive, or after MAXH hours (default 72).
cd /home/infres/yinwang/CS_QMI/ssl_pilot_v9 || exit 1
L=/home/infres/yinwang/CS_QMI/slurm_logs; LOG=$L/v9_n2_watch.log; ST=$L/v9_n2_watch_status.json; MAXH=${MAXH:-72}; t0=$(date +%s)
declare -A state; declare -A label; early_fail=0; ended_fail=0; ended_ok=0
echo "$(date -u +%FT%TZ) watch start pid $$ files: $*" >> "$LOG"
while :; do
  for f in "$@"; do [ -f "$f" ] || continue; while read -r id lab; do [[ "$id" =~ ^[0-9]+$ ]] || continue; [ -z "${state[$id]}" ] && { state[$id]=new; label[$id]="$lab"; }; done < "$f"; done
  all_ended=1
  for id in "${!state[@]}"; do
    [ "${state[$id]}" = ended ] && continue
    q=$(squeue -h -j "$id" -o "%T %M" 2>/dev/null)
    if [ -z "$q" ]; then
      lf=$(ls $L/*_${id}.out 2>/dev/null | head -1); ex=$(grep -o -E "exit=[0-9]+" "$lf" 2>/dev/null | tail -1)
      if [ "$ex" = "exit=0" ]; then ended_ok=$((ended_ok+1)); echo "$(date -u +%FT%TZ) ended ok $id ${label[$id]}" >> "$LOG"
      else ended_fail=$((ended_fail+1)); echo "$(date -u +%FT%TZ) ENDED-NOT-OK $id ${label[$id]} ($ex) $(grep -m1 -E "Traceback|Error|refused|CANCELLED|TIME LIMIT" "$lf" 2>/dev/null | cut -c1-160)" >> "$LOG"; fi
      state[$id]=ended; continue
    fi
    all_ended=0; set -- $q; st=$1; el=$2
    if [ "$st" = RUNNING ] && [ "${state[$id]}" = new ]; then
      secs=$(echo "$el" | awk -F: '{if (NF==3) print $1*3600+$2*60+$3; else if (NF==2) print $1*60+$2; else print 0}' | sed 's/.*-//')
      if [ "${secs:-0}" -ge 60 ]; then
        lf=$(ls $L/*_${id}.out 2>/dev/null | head -1); bad=$(grep -m1 -E "Traceback|refused|Error" "$lf" 2>/dev/null | cut -c1-160)
        if [ -n "$bad" ]; then early_fail=$((early_fail+1)); echo "$(date -u +%FT%TZ) EARLY-FAIL $id ${label[$id]}: $bad" >> "$LOG"; else echo "$(date -u +%FT%TZ) started ok $id ${label[$id]} ($(grep -m1 -o -E "host=[^ ]+ gpu=[^,]+" "$lf"))" >> "$LOG"; fi
        state[$id]=checked
      fi
    fi
  done
  n=${#state[@]}; printf '{"utc":"%s","jobs":%d,"ended_ok":%d,"ended_not_ok":%d,"early_fail":%d,"all_ended":%d,"feeders_alive":%d}\n' "$(date -u +%FT%TZ)" "$n" "$ended_ok" "$ended_fail" "$early_fail" "$all_ended" "$(pgrep -f "slurm/v9/n2_feed.sh|n2_release.py run" | wc -l)" > "$ST"
  if [ "$all_ended" = 1 ] && [ "$n" -gt 0 ] && [ "$(pgrep -f "slurm/v9/n2_feed.sh|n2_release.py run" | wc -l)" = 0 ]; then echo "$(date -u +%FT%TZ) all ended: ok=$ended_ok not_ok=$ended_fail early_fail=$early_fail" >> "$LOG"; break; fi
  [ $(( $(date +%s) - t0 )) -gt $((MAXH*3600)) ] && { echo "$(date -u +%FT%TZ) watch timeout" >> "$LOG"; break; }
  sleep 60
done
