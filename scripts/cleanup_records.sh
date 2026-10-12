#!/usr/bin/env bash
# Owner decision D4 (2026-10-12), groups B–D of reports/OWNER_DECISIONS_20261012.md.  Dry run by default; the owner deletes with
#   DO_DELETE=1 bash scripts/cleanup_records.sh
# B  data/av_mnist_multibench/avmnist/ (my duplicate extraction, train arrays only; no script reads it — checked 2026-10-12, GAIN reads
#    extracted/avmnist/).  Removed only if its three arrays are byte-identical to extracted/avmnist/ and the kept archive zips/avmnist.tar.gz
#    is present with the manifest size (avmnist_temp/ then survives inside that archive).
# C  outputs/P128_* (withdrawn SimCLR partial runs): only the checkpoints (*.pt) go; configs, metrics, logs, sha256 and plots stay as the
#    record.  The 236 untracked reports/**/*.partial.json are kept (owner decision).
# D  finished smoke / check directories: only checkpoint and array files go (*.pt *.pth *.npy *.npz *.safetensors *.bin *.pkl); logs,
#    manifests, JSON / YAML / text records and patches stay in place.  Smoke directories of runs still in development are not listed.
# Every deletion is appended to slurm_logs/cleanup_records.log.
O=/home/infres/yinwang/CS_QMI/outputs; DATA=/projects/EEG-foundation-model/yinghao/data/av_mnist_multibench
LOG=/home/infres/yinwang/CS_QMI/slurm_logs/cleanup_records.log; DEL=${DO_DELETE:-0}; total=0
human() { awk -v b="$1" 'BEGIN {printf "%.2f GB", b / 1e9}'; }
rmfile() { local sz; sz=$(stat -c %s "$1"); total=$((total + sz))
  if [ "$DEL" = 1 ]; then rm -- "$1" && echo "$(date -u +%FT%TZ) deleted $1 $sz" >> "$LOG"; fi; }
echo "== B  $DATA/avmnist"
okB=1
[ "$(stat -c %s "$DATA/zips/avmnist.tar.gz" 2>/dev/null)" = 1600114664 ] || { echo "   kept archive missing or wrong size -> B skipped"; okB=0; }
for f in audio/train_data.npy image/train_data.npy train_labels.npy; do
  [ $okB = 1 ] || break
  if [ -f "$DATA/avmnist/$f" ] && cmp -s "$DATA/avmnist/$f" "$DATA/extracted/avmnist/$f"; then echo "   identical to extracted/: $f"
  else echo "   NOT identical or missing: $f -> B skipped"; okB=0; fi
done
if [ $okB = 1 ] && [ -d "$DATA/avmnist" ]; then szB=$(du -sb "$DATA/avmnist" | cut -f1); total=$((total + szB)); echo "   remove directory: $(human $szB)"
  [ "$DEL" = 1 ] && rm -r -- "$DATA/avmnist" && echo "$(date -u +%FT%TZ) deleted directory $DATA/avmnist $szB" >> "$LOG"; fi
echo "== C  outputs/P128_* checkpoints"
for d in "$O"/P128_*; do [ -d "$d" ] || continue; s0=$total
  while IFS= read -r -d '' f; do rmfile "$f"; done < <(find "$d" -type f -name '*.pt' -print0)
  echo "   $(basename "$d"): $(human $((total - s0)))"; done
echo "== D  smoke / check directories (checkpoints and arrays only)"
for d in R7_D1/d1_30_smoke R7_D1/train_smoke R7_D1/cache_smoke R6_GAIN/smoke R6_REG_cpu_smoke R6_REG_cpu_smoke2 R6_REG_P2_comir_check \
         R6_REG_P2_comir_check_reg R6_REG_P2_cpu_smoke R6_GAIN/superseded_image_roles R7_D1/d1_30_fix_staging R7_D1/d1_30_fix_staging_E2 \
         R7_D1/d1_30_fix_staging_E3 R7_D1/data_provisional; do
  [ -d "$O/$d" ] || { echo "   absent: $d"; continue; }; s0=$total
  while IFS= read -r -d '' f; do rmfile "$f"; done < <(find "$O/$d" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.npy' -o -name '*.npz' \
      -o -name '*.safetensors' -o -name '*.bin' -o -name '*.pkl' \) -print0)
  big=$(find "$O/$d" -type f -size +5M ! \( -name '*.pt' -o -name '*.pth' -o -name '*.npy' -o -name '*.npz' -o -name '*.safetensors' \
      -o -name '*.bin' -o -name '*.pkl' \) | wc -l)
  echo "   $d: $(human $((total - s0)))$([ "$big" -gt 0 ] && echo "  (kept: $big record file(s) > 5 MB)")"; done
echo "$([ "$DEL" = 1 ] && echo deleted || echo 'dry run (nothing deleted)'): $(human $total)"
