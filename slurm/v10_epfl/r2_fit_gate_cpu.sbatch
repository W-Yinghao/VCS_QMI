#!/usr/bin/env bash
#SBATCH --job-name=v10e_r2fitc
#SBATCH --partition=cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V10 EPFL RuLSIF_exact_v2 FIT-only functional gate (amended level A of A1) on the 334 EPFL FIT windows, CPU variant of r2_fit_gate.sbatch
# (same script, --device cpu).  Expected 1.5-3 h (1025 coefficients); per-candidate results are kept, a re-run of the same commit reuses them.
#   S=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_epfl_rulsif2_gate3
#   V10_EPFL_SOURCE=$S V10_EPFL_EXPECT_COMMIT=$(git -C $S rev-parse HEAD) \
#   V10_EPFL_RECORD_SHA256=$(sha256sum $S/v10/epfl/EPFL_ACTIVATION_RECORD.json | cut -c1-64) \
#   V10_EPFL_GROUP_SHA256=$(sha256sum $S/v10/epfl/EPFL_GROUP_RECORD_V2_FROZEN.json | cut -c1-64) \
#     sbatch --export=ALL --job-name=v10e_r2fitc<T> $S/slurm/v10_epfl/r2_fit_gate_cpu.sbatch --centres <T>
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export V10_EPFL_OUT=${V10_EPFL_OUT:-/home/infres/yinwang/CS_QMI/outputs/v10_epfl_rulsif_v2}
: "${SLURM_JOB_ID:?Slurm required}"
: "${V10_EPFL_SOURCE:?frozen gate source required}"
: "${V10_EPFL_EXPECT_COMMIT:?commit pin required}"
: "${V10_EPFL_RECORD_SHA256:?GATE_CHECKS activation record pin required}"
: "${V10_EPFL_GROUP_SHA256:?group record pin required}"
R=$V10_EPFL_SOURCE
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
echo "host=$(hostname) start=$(date -u +%FT%TZ) job=${SLURM_JOB_ID} commit=$(git -C "$R" rev-parse HEAD) expect=${V10_EPFL_EXPECT_COMMIT} record_pin=${V10_EPFL_RECORD_SHA256} group_pin=${V10_EPFL_GROUP_SHA256} out=${V10_EPFL_OUT} threads=${SLURM_CPUS_PER_TASK}"
mkdir -p "$V10_EPFL_OUT"
cd "$R"
set +e
"$PYTHON" -I scripts/v10_epfl/rulsif_v2_fit_gate.py --device cpu "$@"; rc=$?
echo "end=$(date -u +%FT%TZ) exit=${rc}"
exit "$rc"
