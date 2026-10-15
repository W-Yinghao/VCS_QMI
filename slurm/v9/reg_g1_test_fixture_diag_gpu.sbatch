#!/usr/bin/env bash
#SBATCH --job-name=v9_g1test_diag
#SBATCH --time=00:45:00
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nice=300
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Non-test diagnostic of the V9-G1 test-read fixture: run-to-run repeatability of REG-P2 registration for the predictors that did not reproduce their DEV records.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; R=/home/infres/yinwang/CS_QMI/ssl_pilot_v9; cd "$R" || exit 2; export PYTHONPATH="$R/src:$R"; unset V9_REG_ROOT V9_REG_TEST_ROOT V9_REG_TEST_FIXTURE_ROOT
echo "host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git -C "$R" rev-parse --short HEAD)"
"$PYTHON" scripts/v9/reg_g1_test_fixture_diag.py 2>&1 | grep -v ResourceWarning | grep -v "^  "; rc=${PIPESTATUS[0]}; echo "diag_rc=$rc end=$(date -u +%FT%TZ)"; exit $rc
