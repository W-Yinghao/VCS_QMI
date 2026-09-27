#!/usr/bin/env bash
#SBATCH --job-name=precheck_d_cases
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/precheck_d_cases_%x_%j.out
# Pre-check D tests for a subset of cases (features must exist). FEATURES, CASES (comma list), REPORT (prefix, case names appended) required.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
: "${FEATURES:?}"; : "${CASES:?}"; : "${REPORT:?}"
banner "pre-check D cases ${CASES} <- ${FEATURES}"
"$PYTHON" scripts/precheck_d_tests.py --features "$FEATURES" --out "${REPORT}_$(echo "$CASES" | tr ',' '+')" --cases "$CASES" ${TEST_ARGS:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
