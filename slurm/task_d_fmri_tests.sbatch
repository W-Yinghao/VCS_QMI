#!/usr/bin/env bash
#SBATCH --job-name=fmri_tests
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/task_d_fmri_tests_%x_%j.out
# P65 step 2: circular-shift-calibrated dependence tests on the prepared fMRI features. PREP (dir from step 1) and REPORT (prefix) required;
# TEST_ARGS passed through (pass variables via the environment; never commas inside --export values).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${PREP:?}"; : "${REPORT:?}"
banner "P65 fMRI tests: $PREP -> $REPORT (${TEST_ARGS:-})"
"$PYTHON" scripts/task_d_fmri_tests.py --prep "$PREP" --out "$REPORT" ${TEST_ARGS:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
