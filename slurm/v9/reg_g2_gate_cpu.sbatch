#!/usr/bin/env bash
#SBATCH --job-name=v9_reg_g2_gate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-G2 CPU gate: unit tests (G2 + G1 + the verbatim r6 suites), the unit table, the MI / NMI reuse check, then ONE smoke unit (VCS, 200 steps,
# 1 perturbation per window) end to end on the REG-P3-shaped smoke root (4 FIT / 2 CAL / 2 relabelled-CAL "DEV" windows; no DEV window read).
# The other methods' code-path smokes and the cost measurement run in the GPU timing job (G1 lesson: a critic smoke with 2 perturbations took 68 min on CPU).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v9; cd "$R" || exit 2; export PYTHONPATH="$R/src:$R"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export V9_REG_EPFL_ROOT=${V9_REG_EPFL_ROOT:-/projects/EEG-foundation-model/yinghao/outputs_v9/reg_epfl}
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git -C "$R" rev-parse --short HEAD) root=$V9_REG_EPFL_ROOT"
export V9_REG_ROOT=$(mktemp -d)                                                  # the G1 test suite's root (never the live G1 root)
"$PYTHON" -m unittest tests/v9/test_reg_g2.py -v 2>&1 | tail -30; rc1=${PIPESTATUS[0]}
"$PYTHON" -m unittest tests/v9/test_reg.py 2>&1 | tail -4; rc1b=${PIPESTATUS[0]}; [ $rc1b -ne 0 ] && rc1=$rc1b; unset V9_REG_ROOT
"$PYTHON" scripts/v9/reg_g2.py units --csv "$R/configs/V9_REG_EPFL_UNITS.csv"; rc2=$?
"$PYTHON" scripts/v9/reg_g2.py reuse-check | head -20
t0=$(date +%s); "$PYTHON" scripts/v9/reg_g2.py run --unit V9-G2-vcs-frac0.25-eps0.2-s901 --smoke --n-pert 1; rc3=$?; echo "[smoke-exit] V9-G2-vcs-frac0.25-eps0.2-s901 rc=$rc3 $(( $(date +%s) - t0 )) s"
echo "tests_rc=$rc1 units_rc=$rc2 smoke_rc=$rc3 end=$(date -u +%FT%TZ)"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ]
