#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-N2R32 steps 2-3, CPU: V9_SERVER_CN.md section 5 verbatim -- per seed residual_pilot.py fit (8 fixed candidates,
# 24 fits over three seeds) and export under trace_open.py --stage c0, then decide_residual.py.
# A finished fit/export is skipped (fit itself also resumes from its 50-step Adam checkpoints); the decision runs
# only when all three exports exist.  A retry needs a new ATTEMPT label (trace files are never replaced).
# usage: ATTEMPT=attempt1 sbatch [opts] fit_export.sbatch
set -u
source /home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2r32/v9_n2r32/server/env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
A=${ATTEMPT:-attempt1}; C0=$N2R_ROOT/c0; T=$N2R_ROOT/traces
job_record start "attempt=$A"
traced() { local name=$1; shift
  "$PY_CPU" -I -B "$V9_PKG/trace_open.py" --stage c0 --out "$T/${name}_$A.json" -- "$@"
  local r=$?; echo "$name rc=$r $(date -u +%FT%TZ)"; return $r; }
rc=0
for seed in 901 902 903; do
  if [ ! -f "$C0/s$seed/fit/FIT_SELECTION.json" ]; then
    traced fit_s$seed "$V9_PKG/residual_pilot.py" fit --cache "$N2P_AFFINE_ROOT/c0/s$seed/cache" \
      --affine-selection "$N2P_AFFINE_ROOT/c0/s$seed/fit/FIT_SELECTION.json" --out "$C0/s$seed/fit" || { rc=4; continue; }
  fi
  if [ ! -f "$C0/s$seed/export/EXPORT_SUMMARY.json" ]; then
    traced export_s$seed "$V9_PKG/residual_pilot.py" export --cache "$N2P_AFFINE_ROOT/c0/s$seed/cache" \
      --selection "$C0/s$seed/fit/FIT_SELECTION.json" --out "$C0/s$seed/export" || { rc=4; continue; }
  fi
done
if [ $rc -eq 0 ]; then
  traced decision "$V9_PKG/decide_residual.py" \
    --selection "$C0/s901/fit/FIT_SELECTION.json" --selection "$C0/s902/fit/FIT_SELECTION.json" \
    --selection "$C0/s903/fit/FIT_SELECTION.json" --export "$C0/s901/export/EXPORT_SUMMARY.json" \
    --export "$C0/s902/export/EXPORT_SUMMARY.json" --export "$C0/s903/export/EXPORT_SUMMARY.json" \
    --out "$C0/RESIDUAL_DECISION.json" || rc=5
fi
job_record end "attempt=$A rc=$rc"; echo "end=$(date -u +%FT%TZ) rc=$rc"; exit $rc
