#!/usr/bin/env bash
#SBATCH --job-name=v10e_psmoke
#SBATCH --partition=cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# CPU checks of the EPFL production runner: the CPU tests (guard, host / geometry, production contract) and the runner's smoke (every READY
# production family through CAL selection, a final seed with DEV at its selected checkpoint, reuse verification and a resume with log
# truncation) on smoke roles (4 FIT, 2 CAL, 2 CAL windows relabelled DEV: no DEV window is read) with a reduced search.  Not a result.
#   cd /home/infres/yinwang/CS_QMI/ssl_pilot_v10_epfl_prod && V10_EPFL_EXPECT_COMMIT=$(git rev-parse HEAD) sbatch --export=ALL slurm/v10_epfl/prod_smoke_cpu.sbatch
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 V10_TEST_DEVICE=cpu CUDA_VISIBLE_DEVICES=
: "${V10_EPFL_EXPECT_COMMIT:?commit pin required}"
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_epfl_prod
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
commit=$(git -C "$R" rev-parse HEAD); dirty=$(git -C "$R" status --porcelain -- src scripts slurm tests v10/epfl | wc -l)
echo "host=$(hostname) start=$(date -u +%FT%TZ) job=${SLURM_JOB_ID:-NA} commit=${commit} expect=${V10_EPFL_EXPECT_COMMIT} dirty=${dirty}"
if [ "$dirty" -ne 0 ] || [ "$commit" != "$V10_EPFL_EXPECT_COMMIT" ]; then echo "[v10e] REFUSED: dirty tree or commit pin mismatch"; exit 1; fi
cd "$R"
OUT=/home/infres/yinwang/CS_QMI/outputs/v10_epfl_prod_smoke/${SLURM_JOB_ID}   # home (31 TB free); /projects is below its 1.2 TB floor
set +e
PYTHONPATH="$R/src" "$PYTHON" -m pytest -q -p no:cacheprovider -rA --durations=10 tests/v10/test_epfl_guard.py tests/v10/test_epfl_geometry.py tests/v10/test_epfl_production.py; rc1=$?
echo "pytest exit=${rc1} $(date -u +%FT%TZ)"
"$PYTHON" -I scripts/v10_epfl/run.py smoke --device cpu --out "$OUT"; rc2=$?
echo "runner smoke exit=${rc2} $(date -u +%FT%TZ) out=${OUT}"
echo "end=$(date -u +%FT%TZ) exit=${rc1}/${rc2}"
[ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ]
