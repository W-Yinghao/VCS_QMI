#!/usr/bin/env bash
#SBATCH --job-name=precheck_d
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/precheck_d_%x_%j.out
# Pre-check D (leakage detection power) on the CPU partition. RUN, CKPT, OUT (feature dir), REPORT (prefix) required; SMOKE=1 for the probe.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
: "${RUN:?}"; : "${CKPT:?}"; : "${OUT:?}"; : "${REPORT:?}"
banner "pre-check D: features + tests for ${RUN}/${CKPT} (smoke=${SMOKE:-0})"
if [ "${SMOKE:-0}" = "1" ]; then STR="0,0.1"; else STR="${STRENGTHS:-0,0.01,0.02,0.05,0.1,0.2}"; fi
if [ ! -f "$OUT/manifest.json" ]; then
  "$PYTHON" scripts/precheck_d_features.py --run "$RUN" --checkpoint "$CKPT" --out "$OUT" --strengths "$STR" --workers 6 || exit $?
else
  echo "[precheck_d] features exist in $OUT; skipping extraction"
fi
if [ "${SMOKE:-0}" = "1" ]; then
  "$PYTHON" scripts/precheck_d_tests.py --features "$OUT" --out "$REPORT" --smoke; rc=$?
else
  "$PYTHON" scripts/precheck_d_tests.py --features "$OUT" --out "$REPORT" ${TEST_ARGS:-}; rc=$?
fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
