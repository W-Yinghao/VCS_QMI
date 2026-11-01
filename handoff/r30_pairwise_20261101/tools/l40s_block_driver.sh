#!/bin/bash
# One-shot driver for the R30 L40S block (f60_s1002 / f60_s1003), main session 2026-11-01.
# Waits for the 4 L40S parent jobs, binds parents, submits the stage/control lines (no coefficient needed),
# submits the 2 H-parent calibrations, waits, merges coefficients, then submits the PN/PV/PB A1 lines.
# Stops at the first error (set -e); every submission is appended to SUBMITTED.tsv. No --dependency.
set -euo pipefail
D=/home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r30_pairwise_20261101
PKG=/home/infres/yinwang/CS_QMI/r30_v11/VCS_R30_v1_1_Merged_Server_20261101
cd $D
log() { echo "$(date -u +%FT%TZ) $*"; }
submit_line() {  # $1 = full sbatch line, $2 = note
  local n; n=$(echo "$1" | grep -oE 'job-name=[^ ]+' | cut -d= -f2)
  if squeue -u "$USER" -h -o %j | grep -qx "$n"; then log "already queued $n"; return 0; fi
  local out j; out=$(eval "$1" 2>&1) || { log "SUBMIT FAILED $n: $out"; return 1; }
  j=$(echo "$out" | grep -oE '[0-9]{6,}' | tail -1)
  printf "%s\tformal\t%s\t%s\t%s\n" "$(date -u +%FT%TZ)" "$n" "$j" "$2" >> SUBMITTED.tsv; log "OK $n $j"
}
wait_jobs() { while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 120; done; }
log "waiting for L40S parents 1080170-1080173"
wait_jobs 1080170,1080171,1080172,1080173
for p in wp_f60_s1002_H_e5 wp_f60_s1002_L_e5 wp_f60_s1003_H_e5 wp_f60_s1003_L_e5; do
  ls $O/parents/$p/ >/dev/null || { log "MISSING parent dir $p"; exit 1; }
done
/usr/bin/python3 -B tools/bind_parents.py $O $D $PKG
log "parents bound"
grep L40S LINES_R30.txt | grep -vE 'r30a1_f60_s100[23]_P[NVB]$' > LINES_R30_L40S_STAGE.txt
while read -r line; do submit_line "$line" "L40S block stage/control; parents bound by tools/bind_parents.py"; done < LINES_R30_L40S_STAGE.txt
C=()
for p in wp_f60_s1002_H_e5 wp_f60_s1003_H_e5; do
  out=$(sbatch --job-name=r30v11_calib_$p --partition=L40S --exclude=node51,node53,node60 sbatch/r30v11_calib.sbatch $p)
  j=$(echo "$out" | grep -oE '[0-9]{6,}' | tail -1); C+=($j)
  printf "%s\tformal_calib\t%s\t%s (L40S, normal, nice 0; main-session driver)\n" "$(date -u +%FT%TZ)" "$j" "$p" >> SUBMITTED.tsv; log "calib $p $j"
done
wait_jobs "$(IFS=,; echo "${C[*]}")"
/usr/bin/python3 -B tools/merge_coefficients.py $O $D
log "coefficients merged"
grep L40S LINES_R30.txt | grep -E 'r30a1_f60_s100[23]_P[NVB]$' > LINES_R30_L40S_A1.txt
while read -r line; do submit_line "$line" "L40S block A1 seed confirmation; own calibration merged"; done < LINES_R30_L40S_A1.txt
log "L40S block DONE"
