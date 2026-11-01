#!/bin/bash
# Continuation of l40s_block_driver.sh after it stopped on QOSMaxSubmitJobPerUserLimit (main session 2026-11-01).
# Parents are already bound. Order: the 2 H-parent calibrations (short; the A1 units need them), the remaining
# stage/control lines, then merge coefficients and the 6 A1 PN/PV/PB lines. Submit-limit refusals are retried every
# 3 min; any other sbatch error or tool failure stops the script. Skips lines whose job name is already queued or
# already recorded with a job id in SUBMITTED.tsv. No --dependency.
set -euo pipefail
D=/home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r30_pairwise_20261101
cd $D
log() { echo "$(date -u +%FT%TZ) $*"; }
recorded() { awk -F'\t' -v n="$1" '$3==n && $4 ~ /^[0-9]+$/ {f=1} END {exit !f}' SUBMITTED.tsv; }
sb() {  # $1 = job name, rest = sbatch args; echoes job id
  local n=$1; shift; local out
  while :; do
    if out=$(sbatch --parsable "$@" 2>&1); then echo "$out" | grep -oE '^[0-9]+'; return 0; fi
    if echo "$out" | grep -q QOSMaxSubmitJobPerUser; then log "submit limit, retry $n in 180 s" >&2; sleep 180; continue; fi
    log "SUBMIT FAILED $n: $out" >&2; return 1
  done
}
submit_line() {  # $1 = sbatch line from LINES_R30.txt
  local n; n=$(echo "$1" | grep -oE 'job-name=[^ ]+' | cut -d= -f2)
  if squeue -u "$USER" -h -o %j | grep -qx "$n" || recorded "$n"; then log "skip $n (queued or recorded)"; return 0; fi
  local args; args=$(echo "$1" | sed 's/^sbatch //')
  local j; j=$(eval sb "$n" $args)
  printf "%s\tformal\t%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$n" "$j" "L40S block (driver2)" >> SUBMITTED.tsv; log "OK $n $j"
}
wait_jobs() { while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 120; done; }
C=()
for p in wp_f60_s1002_H_e5 wp_f60_s1003_H_e5; do
  j=$(sb r30v11_calib_$p --job-name=r30v11_calib_$p --partition=L40S --exclude=node51,node53,node60 sbatch/r30v11_calib.sbatch $p)
  C+=($j); printf "%s\tformal_calib\t%s\t%s (L40S, normal, nice 0; main-session driver2)\n" "$(date -u +%FT%TZ)" "$j" "$p" >> SUBMITTED.tsv; log "calib $p $j"
done
grep L40S LINES_R30.txt | grep -vE 'r30a1_f60_s100[23]_P[NVB]$' > LINES_R30_L40S_STAGE.txt
while read -r line; do submit_line "$line"; done < LINES_R30_L40S_STAGE.txt
wait_jobs "$(IFS=,; echo "${C[*]}")"
/usr/bin/python3 -B tools/merge_coefficients.py $O $D
log "coefficients merged"
grep L40S LINES_R30.txt | grep -E 'r30a1_f60_s100[23]_P[NVB]$' > LINES_R30_L40S_A1.txt
while read -r line; do submit_line "$line"; done < LINES_R30_L40S_A1.txt
log "L40S block DONE"
