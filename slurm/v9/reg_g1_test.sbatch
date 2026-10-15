#!/usr/bin/env bash
#SBATCH --job-name=v9_g1test
#SBATCH --time=06:00:00
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nice=300
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-G1 sealed-test read, grouped GPU job: REG-P2 registration of the 134 test cores × 10 frozen perturbations for each given predictor in turn
# (the G1 CAL choice as is; a predictor whose result exists is skipped; results are never overwritten).
#   sbatch --job-name=<name> --time=<hh:mm:ss> slurm/v9/reg_g1_test.sbatch <predictor_id> [<predictor_id> ...]
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v9; cd "$R" || exit 2; export PYTHONPATH="$R/src:$R"; unset V9_REG_ROOT V9_REG_TEST_ROOT V9_REG_TEST_FIXTURE_ROOT
echo "host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) start=$(date -u +%FT%TZ) commit=$(git -C "$R" rev-parse --short HEAD) code=$("$PYTHON" -c "import sys; sys.path.insert(0, \"src\"); from v9.reg import g1_test as T; print(T.code_hash())" 2>/dev/null)"
rc_all=0
for p in "$@"; do
  echo "[predictor] $(date -u +%FT%TZ) $p"; "$PYTHON" scripts/v9/reg_g1_test.py run --predictor "$p" 2>&1 | grep -v ResourceWarning | grep -v "^  "; rc=${PIPESTATUS[0]}; echo "[predictor-exit] $p rc=$rc $(date -u +%FT%TZ)"; [ $rc -ne 0 ] && rc_all=$rc
done
echo "exit=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
