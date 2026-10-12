#!/usr/bin/env bash
# Owner decision D4 (2026-10-12): delete only the VERIFIED, re-downloadable archives listed in data/DATA_MANIFEST.json → deletion_candidates
# (recommendation "candidate" and verified_complete true).  Archives marked "keep" (Google-Drive-hosted, QVHighlights mirror) are never touched.
# Each file must still exist with the recorded size.  Dry run by default; the owner deletes with:  DO_DELETE=1 bash scripts/cleanup_data_archives.sh
# Every deletion is appended to slurm_logs/cleanup_data_archives.log (path, bytes, time); archive sha256 stay recorded in the manifest.
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 1; LOG=/home/infres/yinwang/CS_QMI/slurm_logs/cleanup_data_archives.log
mapfile -t ROWS < <(python3 -I -c "
import json; m = json.load(open('data/DATA_MANIFEST.json'))
for a in m['deletion_candidates']['archives']:
    if a.get('recommendation') == 'candidate' and a.get('verified_complete') is True:
        print(a['archive'] + '\t' + str(a['bytes']))")
total=0; n=0; bad=0
for r in "${ROWS[@]}"; do f=${r%%$'\t'*}; b=${r##*$'\t'}
  if [ ! -f "$f" ]; then echo "absent (already gone): $f"; continue; fi
  sz=$(stat -c %s "$f"); if [ "$sz" != "$b" ]; then echo "SIZE MISMATCH, skipped: $f ($sz vs $b)"; bad=$((bad+1)); continue; fi
  n=$((n+1)); total=$((total+sz))
  if [ "${DO_DELETE:-0}" = 1 ]; then rm -- "$f" && echo "$(date -u +%FT%TZ) deleted $f $sz" >> "$LOG" && echo "deleted $f"; else printf '%7.2f GB  %s\n' "$(echo "$sz/1000000000" | bc -l)" "$f"; fi
done
printf '%s: %d archives, %.1f GB%s\n' "$([ "${DO_DELETE:-0}" = 1 ] && echo deleted || echo 'dry run (nothing deleted)')" $n "$(echo "$total/1000000000" | bc -l)" "$([ $bad -gt 0 ] && echo ", $bad skipped on size mismatch")"
