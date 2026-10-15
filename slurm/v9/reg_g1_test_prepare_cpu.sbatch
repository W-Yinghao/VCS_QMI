#!/usr/bin/env bash
#SBATCH --job-name=v9_g1test_prepare
#SBATCH --time=01:00:00
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --nice=300
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-G1 sealed-test read: THE READ STARTS HERE (ledger read_utc).  Archive + per-file sha256 against the frozen list, extraction, REG-P1 cmd_prepare
# verbatim into outputs_v9/reg_test (role TEST), manifest re-checked.  Refuses a prepared root.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v9; cd "$R" || exit 2; export PYTHONPATH="$R/src:$R"; unset V9_REG_ROOT V9_REG_TEST_ROOT V9_REG_TEST_FIXTURE_ROOT
echo "host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) start=$(date -u +%FT%TZ) commit=$(git -C "$R" rev-parse --short HEAD) code=$("$PYTHON" -c "import sys; sys.path.insert(0, \"src\"); from v9.reg import g1_test as T; print(T.code_hash())" 2>/dev/null)"
"$PYTHON" scripts/v9/reg_g1_test.py prepare 2>&1 | grep -v ResourceWarning | tail -5; rc=${PIPESTATUS[0]}
echo "prepare_rc=$rc end=$(date -u +%FT%TZ)"; exit $rc
