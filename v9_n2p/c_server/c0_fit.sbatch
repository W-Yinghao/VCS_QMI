#!/usr/bin/env bash
#SBATCH --job-name=n2pc_fit
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9 N2P C0, CPU: per seed the package's rich_pilot.py validate-cache / fit (16 frozen candidates: scalar/rich x
# VCS/BCE x 2 LR x 2 inits, 400 NumPy fp64 Adam steps; 48 fits over three seeds) / export --include-native-sensitivity,
# student_handoff.py for rich_vcs and rich_bce (original reader; fails by design for a non-deployable posterior),
# then tools/decide_rich_pilot.py -> PILOT_DECISION.json.  Unchanged package scripts under trace_open.py (stage c0).
# Single-threaded BLAS as in the package README.  Finished steps are skipped, nothing is replaced.
set -u
source /home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2p_c/v9_n2p/c_server/env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CACHE=${CACHE:-cache}; J=${SLURM_JOB_ID:-nojob}; OUT=$N2PC_ROOT/c0; RI=$N2PC_PKG/v9/rich_input
mkdir -p "$OUT/traces" "$N2PC_RECORDS/c0/traces"
step() { local name=$1; shift
  "$N2PC_PY_C0" -I -B "$N2PC_SERVER/trace_open.py" --stage c0 --out "$OUT/traces/${name}_$J.json" -- "$@"
  local r=$?; echo "$name rc=$r $(date -u +%FT%TZ)"; return $r; }
echo "host=$(hostname) job=$J start=$(date -u +%FT%TZ) root=$N2PC_ROOT"
rc=0; SEL=(); EXP=()
for s in 901 902 903; do
  D=$OUT/s$s; C=$N2PC_OLD/s$s
  [ -f "$D/$CACHE/CACHE_MANIFEST.json" ] || { echo "s$s: no $CACHE (extraction missing or failed)"; rc=3; continue; }
  if [ ! -f "$D/fit/FIT_SELECTION.json" ]; then
    step validate_s$s "$RI/rich_pilot.py" validate-cache --cache "$D/$CACHE" || { rc=4; continue; }
    step fit_s$s "$RI/rich_pilot.py" fit --cache "$D/$CACHE" --config "$RI/config.json" --out "$D/fit" || { rc=4; continue; }
  fi
  if [ ! -f "$D/export/EXPORT_SUMMARY.json" ]; then
    step export_s$s "$RI/rich_pilot.py" export --cache "$D/$CACHE" --selection "$D/fit/FIT_SELECTION.json" \
      --out "$D/export" --include-native-sensitivity || { rc=4; continue; }
  fi
  SEL+=(--selection "$D/fit/FIT_SELECTION.json"); EXP+=(--export "$D/export/EXPORT_SUMMARY.json")
  for m in rich_vcs rich_bce; do
    M=$(echo "$m" | tr a-z A-Z)
    [ -f "$D/${M}_HANDOFF.json" ] || step handoff_${m}_s$s "$RI/student_handoff.py" --repo "$N2PC_REPO" --cache "$D/$CACHE" \
      --selection "$D/fit/FIT_SELECTION.json" --export-summary "$D/export/EXPORT_SUMMARY.json" \
      --observed "$C/pairs.observed_pairs.json" --method "$m" --out "$D/${M}_HANDOFF.json"
  done
  mkdir -p "$N2PC_RECORDS/c0/s$s"
  cp "$D/fit/FIT_SELECTION.json" "$D/export/EXPORT_SUMMARY.json" "$D"/export/*.lock.json "$N2PC_RECORDS/c0/s$s/"
  cp "$D"/*_HANDOFF.json "$N2PC_RECORDS/c0/s$s/" 2>/dev/null
done
if [ ${#SEL[@]} -gt 0 ]; then
  step decision "$N2PC_PKG/tools/decide_rich_pilot.py" "${SEL[@]}" "${EXP[@]}" --out "$OUT/PILOT_DECISION.json" || rc=5
  cp "$OUT/PILOT_DECISION.json" "$N2PC_RECORDS/c0/" 2>/dev/null
fi
cp "$OUT"/traces/*_"$J".json "$N2PC_RECORDS/c0/traces/" 2>/dev/null
echo "end=$(date -u +%FT%TZ) rc=$rc"; exit $rc
