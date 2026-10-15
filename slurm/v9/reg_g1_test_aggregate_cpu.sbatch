#!/usr/bin/env bash
#SBATCH --job-name=v9_g1test_aggregate
#SBATCH --time=00:30:00
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --nice=300
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-G1 sealed-test read: g1_aggregate.main verbatim over the 54 test results -> reports/V9/V9_G1_TEST_results/
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v9; cd "$R" || exit 2; export PYTHONPATH="$R/src:$R"; unset V9_REG_ROOT V9_REG_TEST_ROOT V9_REG_TEST_FIXTURE_ROOT
echo "host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) start=$(date -u +%FT%TZ) commit=$(git -C "$R" rev-parse --short HEAD) code=$("$PYTHON" -c "import sys; sys.path.insert(0, \"src\"); from v9.reg import g1_test as T; print(T.code_hash())" 2>/dev/null)"
"$PYTHON" scripts/v9/reg_g1_test.py aggregate 2>&1 | grep -v ResourceWarning | tail -12; rc=${PIPESTATUS[0]}
echo "aggregate_rc=$rc end=$(date -u +%FT%TZ)"; exit $rc
