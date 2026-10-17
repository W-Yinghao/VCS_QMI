#!/usr/bin/env bash
#SBATCH --job-name=v10_t01_cpu
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V10 - T01 on CPU (no GPU): package core/verify_torch.py --device cpu (unchanged), design-tolerance core checks (+ NaN /
# constant-feature / dense-QP oracle), real-host adapter acceptance on a real BF-SHG FIT batch (fp64: 4 scenes, fp32: 16 scenes;
# host-init and sharpened feature states; the M1 relative zero-head rule rows) and on the real warp chain (one FIT core with
# REG-P2's perturbation rule), fp64 finite-difference geometric gradient check, host replay (G.cmd_train 2 steps vs the V10
# trainer); merge -> v10/T01.  Reads FIT only (never TEST / CAL / DEV); writes outputs_v10/T01 (and outputs_v10/T02 in mode
# phase1) and v10/T01 (v10/T02).
# Mode (first script argument, default m1_relative):
#   m1_relative  owner decision 2026-10-17 2.3: every T01 CPU part under the new code identity + merge; the T02 CPU head table
#                (package benchmark_torch.py, no M1 path) is not affected and stays the job-1043668 measurement
#   phase1       the phase-1 sequence: the T01 CPU parts + merge, then the T02 CPU head table -> v10/T02/T02_*_cpu.*
#   slurm/v10/submit.sh slurm/v10/t01_cpu.sbatch -- m1_relative     (never plain sbatch: the header requires the submission pin)
set -o pipefail
MODE=${1:-m1_relative}
case "$MODE" in
  m1_relative|phase1) ;;
  *) echo "[v10] REFUSED: unknown mode '$MODE' (m1_relative | phase1)"; exit 2 ;;
esac
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10
cd "$R" || exit 2
export PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python PYTHONUNBUFFERED=1 PYTHONPATH="$R/src:$R"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} CUBLAS_WORKSPACE_CONFIG=:4096:8
export V10_OUT_ROOT=/projects/EEG-foundation-model/yinghao/outputs_v10
source "$R/slurm/v10/v10_header.sh" || exit 2
v10_header cpu slurm/v10/t01_cpu.sbatch "$MODE" || exit 3
mkdir -p "$V10_OUT_ROOT/T01" "$V10_OUT_ROOT/T02"
"$PYTHON" -m pytest tests/v10 -q -p no:cacheprovider 2>&1 | tail -4; echo "[v10] unit tests rc=${PIPESTATUS[0]} (recorded, not gating)"
"$PYTHON" -W ignore::UserWarning scripts/v10/t01_validate.py run --device cpu --parts verify_torch,core_tol,adapter,geometry,host_replay; rc=$?
"$PYTHON" -W ignore::UserWarning scripts/v10/t01_validate.py merge; rc_m=$?
rc2=NOT_RUN
if [ "$MODE" = phase1 ]; then
  "$PYTHON" -W ignore::UserWarning scripts/v10/t02_timing.py run --device cpu --parts head; rc2=$?
fi
echo "[v10] t01_cpu mode=$MODE rc=$rc merge_rc=$rc_m t02_cpu_head_rc=$rc2 end=$(date -u +%FT%TZ)"
exit $rc
