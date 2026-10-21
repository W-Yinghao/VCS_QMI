#!/usr/bin/env bash
#SBATCH --job-name=n2pc_x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9 N2P C0, GPU: the package's torch_crosscheck.py, then for each seed extract_sgr.py --cache-mode new-cache
# on the existing frozen teacher and rich_pilot.py validate-cache.  Package scripts run unchanged, each under
# trace_open.py (stage c0: no DEV/TEST/eval_only/noise files).  Partition/qos/exclude/nice come from the sbatch
# line.  Nothing is replaced: a finished seed is skipped; a failed seed keeps its directory (rerun that seed with
# SEEDS=<seed> CACHE=<new name>, and give the same CACHE to c0_fit.sbatch).
set -u
source /home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2p_c/v9_n2p/c_server/env.sh
CACHE=${CACHE:-cache}; SEEDS=${SEEDS:-901 902 903}; J=${SLURM_JOB_ID:-nojob}
OUT=$N2PC_ROOT/c0; RI=$N2PC_PKG/v9/rich_input; mkdir -p "$OUT/traces" "$N2PC_RECORDS/c0/traces"
T=("$N2PC_PY_C0" -I -B "$N2PC_SERVER/trace_open.py" --stage c0)
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) job=$J start=$(date -u +%FT%TZ) root=$N2PC_ROOT"
rc=0
receipt_pass() { "$N2PC_PY_C0" -I -B -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status") == "PASS" else 1)' "$1" 2>/dev/null; }
# The crosscheck is a CPU-only check; a PASS receipt already produced in the C0 env (records dir) is reused, not rerun.
if [ ! -f "$OUT/TORCH_CROSSCHECK_SERVER.json" ] && receipt_pass "$N2PC_RECORDS/c0/TORCH_CROSSCHECK_SERVER.json"; then
  cp "$N2PC_RECORDS/c0/TORCH_CROSSCHECK_SERVER.json" "$OUT/"; echo "crosscheck: reusing PASS receipt from $N2PC_RECORDS/c0"
fi
if [ -f "$OUT/TORCH_CROSSCHECK_SERVER.json" ]; then
  receipt_pass "$OUT/TORCH_CROSSCHECK_SERVER.json" || { echo "REFUSED: $OUT/TORCH_CROSSCHECK_SERVER.json is not PASS"; exit 3; }
else
  "${T[@]}" --out "$OUT/traces/crosscheck_$J.json" -- "$RI/torch_crosscheck.py" --repo "$N2PC_REPO" \
    --out "$OUT/TORCH_CROSSCHECK_SERVER.json"
  r=$?; echo "crosscheck rc=$r $(date -u +%FT%TZ)"; cp "$OUT/TORCH_CROSSCHECK_SERVER.json" "$N2PC_RECORDS/c0/" 2>/dev/null
  [ $r -eq 0 ] || { cp "$OUT"/traces/*_"$J".json "$N2PC_RECORDS/c0/traces/"; exit $r; }
fi
for s in $SEEDS; do
  C=$N2PC_OLD/s$s; D=$OUT/s$s
  if [ -f "$D/$CACHE/CACHE_MANIFEST.json" ]; then echo "s$s: $CACHE already extracted, skipped"; continue; fi
  "${T[@]}" --out "$OUT/traces/extract_s${s}_$J.json" -- "$RI/extract_sgr.py" --repo "$N2PC_REPO" \
    --teacher "$C/teacher/teacher.pt" --roles "$C/roles.json" --observed "$C/pairs.observed_pairs.json" \
    --reference-u-weights "$C/weights/VCS_Weight.csv" --data-dir "$N2PC_DATA" --vocab "$N2PC_VOCAB" \
    --seed "$s" --device cuda --batch-size 128 --cache-mode new-cache --out "$D/$CACHE"
  r=$?; echo "extract s$s rc=$r $(date -u +%FT%TZ)"; [ $r -eq 0 ] || { rc=$r; continue; }
  "${T[@]}" --out "$OUT/traces/validate_s${s}_$J.json" -- "$RI/rich_pilot.py" validate-cache --cache "$D/$CACHE"
  r=$?; echo "validate s$s rc=$r"; [ $r -eq 0 ] || rc=$r
  mkdir -p "$N2PC_RECORDS/c0/s$s" && cp "$D/$CACHE/CACHE_MANIFEST.json" "$N2PC_RECORDS/c0/s$s/"
done
cp "$OUT"/traces/*_"$J".json "$N2PC_RECORDS/c0/traces/"
echo "end=$(date -u +%FT%TZ) rc=$rc"; exit $rc
