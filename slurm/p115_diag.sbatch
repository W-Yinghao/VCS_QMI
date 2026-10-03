#!/usr/bin/env bash
#SBATCH --job-name=p115_diag
#SBATCH --partition=A100,L40S,RTX6000PRO,H100
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P115 read-only augmentation / gradient diagnostics (v5 §3).  Env: RUNS (comma list, required), OUT_DIR (default outputs/P115_diag_results),
# EXTRA (e.g. "--smoke --checkpoints 20,800"), GATE=1 -> run tests/test_p115_diag.py first and write reports/P115_DIAG_GATE_<job>/.
# One job runs all RUNS sequentially; finished per-run JSONs are skipped.  Normal QOS only.  CPU gate: sbatch -p CPU --gres=gpu:0 ...
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${RUNS:?}"
OUT_DIR=${OUT_DIR:-/home/infres/yinwang/CS_QMI/outputs/P115_diag_results}
banner "P115 diag: runs=$RUNS out=$OUT_DIR extra=${EXTRA:-} gate=${GATE:-0}"
rc=0
if [ "${GATE:-0}" = "1" ]; then
  G=reports/P115_DIAG_GATE_${SLURM_JOB_ID:-local}; mkdir -p "$G"
  "$PYTHON" -m pytest -q tests/test_p115_diag.py 2>&1 | tee "$G/pytest_test_p115_diag.txt" | tail -3; rc=${PIPESTATUS[0]}
  OUT_DIR=$G/smoke
fi
"$PYTHON" scripts/p115_diag.py --runs "$RUNS" --out-dir "$OUT_DIR" ${EXTRA:-} || rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
