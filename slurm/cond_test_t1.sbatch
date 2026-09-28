#!/usr/bin/env bash
#SBATCH --job-name=cond_test_t1
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/cond_test_t1_%x_%j.out
# T1 (next round, T line): within-class conditional test with fair controls, N | Y re-drawn per repeat.
# Env: REPORT (prefix, required); FEATURES_COLOUR and/or FEATURES_BLUR (P45 feature dirs of one encoder); TEST_ARGS passed through
# (pass variables via the environment, never commas inside --export values); SMOKE=1 -> tiny grid (n = 200, R = 5, 20 permutations).
# CPU smoke: sbatch -p CPU --gres=gpu:0 ... (the GPU header is overridden on the command line).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${REPORT:?}"
ARGS=""
[ -n "${FEATURES_COLOUR:-}" ] && ARGS="$ARGS --features-colour $FEATURES_COLOUR"
[ -n "${FEATURES_BLUR:-}" ] && ARGS="$ARGS --features-blur $FEATURES_BLUR"
[ "${SMOKE:-0}" = "1" ] && ARGS="$ARGS --smoke"
banner "T1 conditional test: colour=${FEATURES_COLOUR:-none} blur=${FEATURES_BLUR:-none} -> $REPORT (${TEST_ARGS:-}) smoke=${SMOKE:-0}"
"$PYTHON" scripts/cond_test_t1.py $ARGS --out "$REPORT" ${TEST_ARGS:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
