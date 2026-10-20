#!/usr/bin/env bash
# Read-only Slurm recorder for V9 N2M jobs (sacct/slurmdbd is unreachable on this cluster since 2026-10-15).
# Every 30 s, for each of my jobs whose name starts with v9n2m_ or n2m_ (N2M jobs are named n2m_* so that the running N2F closeout controllers, which wait for every job named v9*, are not blocked), save the latest `scontrol show job -o` line to
# OUT/slurm_records/scontrol_<jobid>.txt (and append it to the job's history file when the line changes).
# A finished job stays visible to scontrol for MinJobAge=300 s, so its final JobState/ExitCode/StartTime/EndTime
# is captured.  Never submits, cancels or modifies a job.  usage: record_slurm.sh OUT_ROOT MAX_HOURS
OUT="$1"; HMAX="${2:-96}"; D="$OUT/slurm_records"; mkdir -p "$D"; t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt $(( HMAX * 3600 )) ]; do
  for id in $(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk '$2 ~ /^(v9n2m_|n2m_)/ {print $1}'; ls "$D" 2>/dev/null | sed -n 's/^scontrol_\([0-9]*\)\.txt$/\1/p'); do
    line=$(scontrol show job -o "$id" 2>/dev/null | head -1)
    [ -z "$line" ] && continue
    if [ "$line" != "$(cat "$D/scontrol_$id.txt" 2>/dev/null)" ]; then
      printf '%s\n' "$line" > "$D/scontrol_$id.txt"; printf '%s %s\n' "$(date -u +%FT%TZ)" "$line" >> "$D/history_$id.txt"
    fi
  done
  sleep 30
done
