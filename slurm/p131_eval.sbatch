#!/usr/bin/env bash
#SBATCH --job-name=p131_eval
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P131 step 2 (GPU, one bundle): frozen-h pilot linear head per run, accuracy on every regenerated corruption x severity; finished runs
# are skipped; then the aggregate.  Env: RUNS (comma list; default all 21 — set it in the environment, NEVER inside --export=ALL,... where
# SLURM splits on commas), OUT, SMOKE_N, AGG_OUT.  Normal QOS only; node51 and node60 (throttled GPUs) excluded.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P131 evaluate: regen=${P131_REGEN_DIR:-$OUTPUT_ROOT/P131_c10c_regen} out=${OUT:-$OUTPUT_ROOT/P131_results} smoke_n=${SMOKE_N:-0}"
ARGS=(--out "${OUT:-$OUTPUT_ROOT/P131_results}" --smoke-n "${SMOKE_N:-0}")
[ -n "${RUNS:-}" ] && ARGS+=(--runs "$RUNS")
"$PYTHON" scripts/p131_c10c.py evaluate "${ARGS[@]}"; rc=$?
if [ $rc -eq 0 ]; then "$PYTHON" scripts/p131_c10c.py aggregate --in "${OUT:-$OUTPUT_ROOT/P131_results}" --out "${AGG_OUT:-reports/P131_c10c_results}"; rc=$?; fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
