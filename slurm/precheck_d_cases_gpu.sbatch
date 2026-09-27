#!/usr/bin/env bash
#SBATCH --job-name=precheck_d_gpu
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/precheck_d_gpu_%x_%j.out
# Pre-check D tests for a subset of cases on one GPU. FEATURES, CASES (comma list), REPORT (prefix) required; TEST_ARGS passed through.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${FEATURES:?}"; : "${CASES:?}"; : "${REPORT:?}"
banner "pre-check D (GPU) cases ${CASES} <- ${FEATURES}"
"$PYTHON" scripts/precheck_d_tests.py --features "$FEATURES" --out "${REPORT}_$(echo "$CASES" | tr ',' '+')" --cases "$CASES" ${TEST_ARGS:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
