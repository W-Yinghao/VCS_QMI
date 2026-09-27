#!/usr/bin/env bash
#SBATCH --job-name=task_d_shortcut
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/task_d_shortcut_%x_%j.out
# D-T: train one shortcut classifier (RHO, SEED, OUT required) and write its detection features; the conditional tests are separate jobs
# (slurm/precheck_d_cases_gpu.sbatch, one case each).  SMOKE=1 = tiny run + the tests' --smoke on the result; submit the smoke on the CPU
# partition with:  sbatch -p CPU --gres=gpu:0 --cpus-per-task=8 slurm/task_d_shortcut.sbatch
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
: "${RHO:?}"; : "${SEED:?}"; : "${OUT:?}"
# SHORTCUT=colour (default, P55) | blur (P55 addendum 1: Gaussian blur radius 1 px); optional SHIFT overrides the family default.
SC_ARGS="--shortcut ${SHORTCUT:-colour}"; [ -n "${SHIFT:-}" ] && SC_ARGS="$SC_ARGS --shift $SHIFT"
banner "D-T shortcut model rho=${RHO} seed=${SEED} shortcut=${SHORTCUT:-colour} -> ${OUT} (smoke=${SMOKE:-0})"
if [ "${SMOKE:-0}" = "1" ]; then
  "$PYTHON" scripts/task_d_shortcut.py --rho "$RHO" --seed "$SEED" --out "$OUT" $SC_ARGS --workers 4 --smoke ${TRAIN_ARGS:-} || exit $?
  "$PYTHON" scripts/precheck_d_tests.py --features "$OUT" --out "${OUT}/smoke_tests" --smoke; rc=$?
else
  "$PYTHON" scripts/task_d_shortcut.py --rho "$RHO" --seed "$SEED" --out "$OUT" $SC_ARGS --workers 6 ${TRAIN_ARGS:-}; rc=$?
fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
