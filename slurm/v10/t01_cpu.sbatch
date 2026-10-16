#!/usr/bin/env bash
#SBATCH --job-name=v10_t01_cpu
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V10 phase 1 - T01 on CPU (no GPU): package core/verify_torch.py --device cpu (unchanged), design-tolerance core checks, real-host
# adapter acceptance on a real BF-SHG FIT batch (fp64: 4 scenes, fp32: 16 scenes) and on the real warp chain (one FIT core with
# REG-P2's perturbation rule), fp64 finite-difference geometric gradient check; then merge -> v10/T01/TORCH_VALIDATION.json and
# v10/T01/GEOMETRY_GRADIENT_CHECK.json.  Reads FIT only (never TEST / CAL / DEV); writes outputs_v10/T01 and v10/T01 only.
#   sbatch slurm/v10/t01_cpu.sbatch
set -o pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10
cd "$R" || exit 2
export PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python PYTHONUNBUFFERED=1 PYTHONPATH="$R/src:$R"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} CUBLAS_WORKSPACE_CONFIG=:4096:8
export V10_OUT_ROOT=/projects/EEG-foundation-model/yinghao/outputs_v10
source "$R/slurm/v10/v10_header.sh" || exit 2
v10_header cpu || exit 3
mkdir -p "$V10_OUT_ROOT/T01"
"$PYTHON" -m pytest tests/v10 -q -p no:cacheprovider 2>&1 | tail -4; echo "[v10] unit tests rc=${PIPESTATUS[0]} (recorded, not gating)"
"$PYTHON" -W ignore::UserWarning scripts/v10/t01_validate.py run --device cpu --parts verify_torch,core_tol,adapter,geometry; rc=$?
"$PYTHON" -W ignore::UserWarning scripts/v10/t01_validate.py merge; rc_m=$?
echo "[v10] t01_cpu rc=$rc merge_rc=$rc_m end=$(date -u +%FT%TZ)"
exit $rc
