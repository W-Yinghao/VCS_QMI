#!/usr/bin/env bash
#SBATCH --job-name=v10e_smoke
#SBATCH --partition=cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V10 EPFL CPU checks before any GPU submission: (1) the CPU tests (guard, host, geometry), (2) the item-5 gate script in --smoke mode on CPU
# (every part, all 11 families, no full search), (3) the timing script in --smoke mode on CPU (3 steps, one registration per family).
# Code-path checks only (SMOKE_..._NOT_A_GATE_RESULT / SMOKE_OK_NOT_A_COST_RECORD); FIT / CAL windows only.  Submit from the clean tree:
#   cd /home/infres/yinwang/CS_QMI/ssl_pilot_v10_epfl
#   V10_EPFL_EXPECT_COMMIT=$(git rev-parse HEAD) V10_EPFL_RECORD_SHA256=$(sha256sum v10/epfl/EPFL_ACTIVATION_RECORD.json | cut -c1-64) \
#     sbatch --export=ALL slurm/v10_epfl/smoke_cpu.sbatch
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 V10_TEST_DEVICE=cpu CUDA_VISIBLE_DEVICES=
: "${V10_EPFL_EXPECT_COMMIT:?commit pin required}"
: "${V10_EPFL_RECORD_SHA256:?activation record pin required}"
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_epfl
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
echo "host=$(hostname) start=$(date -u +%FT%TZ) job=${SLURM_JOB_ID:-NA} partition=${SLURM_JOB_PARTITION:-NA} commit=$(git -C "$R" rev-parse HEAD) expect=${V10_EPFL_EXPECT_COMMIT}"
cd "$R"
set +e
PYTHONPATH="$R/src" "$PYTHON" -m pytest -q -p no:cacheprovider -rA --durations=10 tests/v10/test_epfl_guard.py tests/v10/test_epfl_geometry.py; rc1=$?
echo "pytest exit=${rc1} $(date -u +%FT%TZ)"
"$PYTHON" -I scripts/v10_epfl/gpu_gate.py --smoke --device cpu; rc2=$?
echo "gate smoke exit=${rc2} $(date -u +%FT%TZ)"
"$PYTHON" -I scripts/v10_epfl/timing.py --smoke --device cpu; rc3=$?
echo "timing smoke exit=${rc3} $(date -u +%FT%TZ)"
echo "end=$(date -u +%FT%TZ) exit=${rc1}/${rc2}/${rc3}"
[ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ] && [ "$rc3" -eq 0 ]
