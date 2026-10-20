#!/usr/bin/env bash
#SBATCH --job-name=v10e_tests
#SBATCH --partition=cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V10 EPFL CPU tests: the data gate (tests/v10/test_epfl_guard.py) and the host / geometry code (tests/v10/test_epfl_geometry.py) on CPU,
# real FIT / CAL windows through the gate at stage GATE_CHECKS (no DEV, no registration error).  Submit from the clean committed tree:
#   cd /home/infres/yinwang/CS_QMI/ssl_pilot_v10_epfl && V10_EPFL_EXPECT_COMMIT=$(git rev-parse HEAD) sbatch --export=ALL slurm/v10_epfl/tests_cpu.sbatch
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 V10_TEST_DEVICE=cpu CUDA_VISIBLE_DEVICES=
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_epfl
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
commit=$(git -C "$R" rev-parse HEAD)
dirty=$(git -C "$R" status --porcelain -- src/v10 scripts/v10_epfl slurm/v10_epfl tests/v10 v10/epfl src/v9/reg | wc -l)
echo "host=$(hostname) start=$(date -u +%FT%TZ) job=${SLURM_JOB_ID:-NA} partition=${SLURM_JOB_PARTITION:-NA} commit=${commit} expect=${V10_EPFL_EXPECT_COMMIT:-UNSET} dirty=${dirty}"
if [ "$dirty" -ne 0 ] || [ "$commit" != "${V10_EPFL_EXPECT_COMMIT:-}" ]; then echo "[v10e] REFUSED: dirty tree or commit pin mismatch"; exit 1; fi
cd "$R"
set +e
PYTHONPATH="$R/src" "$PYTHON" -m pytest -q -p no:cacheprovider -rA --durations=15 tests/v10/test_epfl_guard.py tests/v10/test_epfl_geometry.py
rc=$?
echo "end=$(date -u +%FT%TZ) exit=${rc}"
exit "$rc"
