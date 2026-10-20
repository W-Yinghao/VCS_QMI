#!/usr/bin/env bash
#SBATCH --job-name=v9n2m_cpucheck
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Server-side CPU checks for the V9 N2M package (no GPU, no training, no eval_only labels):
#   1. run_n2m.py inspect: rebuild the inherited N2F 33-unit census from artifacts, incl. best/last checkpoint SHA-256
#      and the full input identity (token cache, observed pools, VCS CSV/locks, vocab, train/DEV size+mtime);
#      exit 2 = NOT_READY while N2F units are still running (expected before 1046654 finishes).
#   2. the package's unit tests (tests/, PYTHONPATH=src) on the server Python.
#   3. scripts/validate_local.py --full-shape (synthetic production-size controls/scoring chain).
# usage: sbatch cpu_checks.sbatch PACKAGE_ROOT BASE_HANDOFF NEW_OUTPUT PYTHON_EXECUTABLE
set -u
P="$1"; B="$2"; O="$3"; PY="$4"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "host=$(hostname) start=$(date -u +%FT%TZ) job=${SLURM_JOB_ID:-} package=$P base=$B out=$O python=$("$PY" --version 2>&1)"
cd "$P" || exit 3
"$PY" "$P/run_n2m.py" verify-package > /dev/null; echo "verify_package_rc=$?"
"$PY" "$P/run_n2m.py" inspect --base "$B" --out "$O/inspection"; echo "inspect_rc=$?"
mkdir -p "$O/package_checks"
PYTHONPATH=src "$PY" -m unittest discover -s tests -v > "$O/package_checks/unittest_output.txt" 2>&1; rc=$?
tail -4 "$O/package_checks/unittest_output.txt"; echo "unittest_rc=$rc"
PYTHONPATH=src "$PY" scripts/validate_local.py --baseline-source "$B/source" --full-shape --out "$O/package_checks/local_validation"; echo "validate_local_rc=$?"
echo "end=$(date -u +%FT%TZ)"
