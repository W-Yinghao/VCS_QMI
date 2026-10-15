#!/usr/bin/env bash
#SBATCH --job-name=v9_g1test_fixture
#SBATCH --time=01:30:00
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nice=300
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-G1 sealed-test read, non-test smoke (guards-need-smoke): every formal function on 3 REG-P1 DEV cores packed into a fixture zip with the test
# naming; cache equality with REG-P1, perturbations equal to REG-P2, all 54 predictors through the formal register path reproducing their G1 / REG-P2
# DEV records, guards (prepare twice, register twice, test root / sealed archive refused), the readings path.  Never opens the sealed archive.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v9; cd "$R" || exit 2; export PYTHONPATH="$R/src:$R"; unset V9_REG_ROOT V9_REG_TEST_ROOT V9_REG_TEST_FIXTURE_ROOT
echo "host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) start=$(date -u +%FT%TZ) commit=$(git -C "$R" rev-parse --short HEAD) code=$("$PYTHON" -c "import sys; sys.path.insert(0, \"src\"); from v9.reg import g1_test as T; print(T.code_hash())" 2>/dev/null)"
"$PYTHON" -m unittest tests/v9/test_reg_g1_test.py 2>&1 | grep -E "^Ran|^OK|^FAILED"; rc1=${PIPESTATUS[0]}
"$PYTHON" scripts/v9/reg_g1_test.py fixture --n-cores 3 --n-pert 1 2>&1 | grep -v ResourceWarning | grep -v "^  " ; rc2=${PIPESTATUS[0]}
echo "tests_rc=$rc1 fixture_rc=$rc2 end=$(date -u +%FT%TZ)"; [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
