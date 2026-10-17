#!/usr/bin/env bash
#SBATCH --job-name=v10_t03_smoke
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V10 T03 smoke (decision 2026-10-17 9.4: one condition x the five families, full 2000-step trainings with the actual full training
# cost measured, before any grid unit; protocol P2 t03.smoke).  CPU only; synthetic data only (no BF-SHG, no TEST, no EPFL).
# Sequence (each step's status explicit in the log; nothing is skipped silently):
#   1  tests/v10_t03 (recorded)                              4  gates G1-G5 (uniform derangements, U identities)  -> gates/
#   2  the 1219-row unit table vs its pin + derivation      5  the 7 smoke units T03_SMK_001..007 (diagonal_mixture m32 rho0.75,
#   3  suite seed 20261016 (generator, hashes) + package        MF perm1 / M0 / M1 cap 10, 30, 100 / MLP perm1 / perm8, lr 3e-4,
#      checker check_synthetic.py on it         -> checks/      algo 1017): train + CAL readout + evaluator    -> runs/
#                                                            6  smoke summary + grid cost projection (v10/tools) -> aggregate/
# Units 5 run only when 2-4 PASS (the runner also refuses / BLOCKs on its own).  Writes outputs_v10/T03 only.
#   slurm/v10_t03/submit.sh slurm/v10_t03/t03_smoke.sbatch      (never plain sbatch: the header requires the submission pin)
set -o pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_t03
cd "$R" || exit 2
export PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python PYTHONUNBUFFERED=1 PYTHONPATH="$R/src:$R/scripts/v10_t03:$R"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} CUBLAS_WORKSPACE_CONFIG=:4096:8
export V10_T03_OUT_ROOT=/projects/EEG-foundation-model/yinghao/outputs_v10/T03
source "$R/slurm/v10_t03/t03_header.sh" || exit 2
t03_header cpu slurm/v10_t03/t03_smoke.sbatch smoke || exit 3
mkdir -p "$V10_T03_OUT_ROOT"
T0=$(date +%s)
"$PYTHON" -m pytest tests/v10_t03 -q -p no:cacheprovider 2>&1 | tail -5; rc_t=${PIPESTATUS[0]}
echo "[v10_t03] unit tests rc=${rc_t} (recorded)"
"$PYTHON" scripts/v10_t03/t03_run.py units --check --output "$V10_T03_OUT_ROOT/checks/T03_UNITS_CHECK.json"; rc_u=$?
"$PYTHON" scripts/v10_t03/t03_run.py gen --root-seed 20261016; rc_g=$?
"$PYTHON" scripts/v10_t03/t03_run.py gate; rc_d=$?
echo "[v10_t03] units_check rc=${rc_u} suite+checker rc=${rc_g} gates rc=${rc_d} t=$(( $(date +%s) - T0 ))s"
rc_s=NOT_RUN
if [ "$rc_u" -eq 0 ] && [ "$rc_g" -eq 0 ] && [ "$rc_d" -eq 0 ]; then
  "$PYTHON" scripts/v10_t03/t03_run.py run --block smoke; rc_s=$?
else
  echo "[v10_t03] smoke units NOT_RUN: a prerequisite failed (units_check / suite+checker / gates)"
fi
"$PYTHON" v10/tools/t03_aggregate.py smoke; rc_a=$?
echo "[v10_t03] t03_smoke tests_rc=${rc_t} units_check_rc=${rc_u} suite_rc=${rc_g} gates_rc=${rc_d} smoke_units_rc=${rc_s} summary_rc=${rc_a} wall=$(( $(date +%s) - T0 ))s end=$(date -u +%FT%TZ)"
[ "$rc_s" = 0 ] && [ "$rc_a" -eq 0 ] || exit 1
