#!/usr/bin/env bash
# Reclaim disk under $OUTPUT_ROOT without touching anything a running / resumable / final result depends on.
#   dry run (default):  bash scripts/cleanup_outputs.sh
#   delete:             DO_DELETE=1 bash scripts/cleanup_outputs.sh
# Rules (all restricted to runs whose status.json says COMPLETED and whose final epoch checkpoint exists):
#   R1  last.pt                       -> deleted (byte-identical copy of the final epoch_XXX.pt for a COMPLETED run)
#   R2  intermediate epoch_XXX.pt     -> deleted only for the superseded search stages (P5 .. P31); kept for P33+ (recipe, ceiling)
#   R3  features/ cache               -> deleted only for P5 .. P31 (regenerated automatically by vcs_ssl.evaluate if ever needed)
#   R4  RESUME_TEST_* run dirs        -> deleted (walltime-chain test artefacts; the evidence is in slurm_logs and job_ids.json)
# Never touched: RUNNING / STOPPED_BUDGET / FAILED runs, initial.pt, final checkpoints, evaluations/, logs/, manifests, configs, reports.
set -euo pipefail
OUT=${OUTPUT_ROOT:-/home/infres/yinwang/CS_QMI/outputs}
PY=/home/infres/yinwang/CS_QMI/env/bin/python
KEEP_INTERMEDIATE_REGEX='^P(3[3-9]|[4-9][0-9])_'   # stages P33 and later keep intermediate checkpoints and feature caches
list=$(mktemp)
for d in "$OUT"/*/; do
  run=$(basename "$d")
  if [[ "$run" == RESUME_TEST_* ]]; then echo "$d" >> "$list"; continue; fi
  [ -f "$d/status.json" ] || continue
  read -r status ep < <("$PY" -c "import json;s=json.load(open('$d/status.json'));print(s.get('status'), s.get('completed_epoch'))")
  [ "$status" = "COMPLETED" ] || continue
  final=$(printf "%s/checkpoints/epoch_%03d.pt" "$d" "$ep")
  [ -f "$final" ] || continue
  [ -f "$d/checkpoints/last.pt" ] && echo "$d/checkpoints/last.pt" >> "$list"                      # R1
  if ! [[ "$run" =~ $KEEP_INTERMEDIATE_REGEX ]]; then
    for c in "$d"/checkpoints/epoch_*.pt; do [ "$c" != "$final" ] && echo "$c" >> "$list"; done      # R2
    [ -d "$d/features" ] && echo "$d/features" >> "$list"                                           # R3
  fi
done
n=$(wc -l < "$list"); total=$(du -shc $(cat "$list") 2>/dev/null | tail -1 | cut -f1)
echo "candidates: $n paths, $total"
awk -F/ '{print $(NF-1)"/"$NF}' "$list" | sed -E 's|^(.*)/(last\.pt)$|R1 last.pt|; s|^(.*)/(epoch_[0-9]+\.pt)$|R2 intermediate ckpt|; s|^(.*)/features$|R3 features|; s|^RESUME_TEST.*|R4 resume test|' | sort | uniq -c
if [ "${DO_DELETE:-0}" = "1" ]; then
  xargs -a "$list" rm -rf --; echo "deleted."
else
  echo "(dry run; re-run with DO_DELETE=1 to delete; full list: $list)"
fi
