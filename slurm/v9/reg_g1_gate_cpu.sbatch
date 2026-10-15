#!/usr/bin/env bash
#SBATCH --job-name=v9_reg_g1_gate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-G1 CPU gate: unit tests (+ the verbatim r6 modules' own suites against the copies), then a smoke of every method on REG-P1's smoke cache
# (4 FIT / 2 CAL / 2 DEV cores; 200 training steps; smoke roots under $V9_REG_ROOT/units/<unit>/{p1,p2}/smoke), then the smoke aggregate.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v9; cd "$R" || exit 2; export PYTHONPATH="$R/src:$R"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export V9_REG_ROOT=${V9_REG_ROOT:-/projects/EEG-foundation-model/yinghao/outputs_v9/reg}
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git -C "$R" rev-parse --short HEAD) root=$V9_REG_ROOT"
"$PYTHON" -m unittest tests/v9/test_reg.py -v 2>&1 | tail -25; rc1=${PIPESTATUS[0]}
"$PYTHON" scripts/v9/reg_g1.py units --csv "$R/configs/V9_REG_UNITS.csv"; rc2=$?
"$PYTHON" scripts/v9/reg_g1.py reuse-check | head -20
rc3=0
for u in V9-G1-vcs-frac0.25-eps0.2-s901 V9-G1-logistic-frac0.25-eps0.2-s901 V9-G1-infonce-frac1-eps0-s901 V9-G1-comir-frac0.25-eps0.2-s901 V9-G1-kernel-frac0.25-eps0.2-s901; do
  t0=$(date +%s); "$PYTHON" scripts/v9/reg_g1.py run --unit "$u" --smoke --n-pert 2; rc=$?; echo "[smoke-exit] $u rc=$rc $(( $(date +%s) - t0 )) s"; [ $rc -ne 0 ] && rc3=$rc
done
"$PYTHON" scripts/v9/reg_g1.py aggregate --smoke --out "$R/reports/V9/V9_G1_GATE/smoke_results" 2>&1 | tail -15; rc4=$?
echo "tests_rc=$rc1 units_rc=$rc2 smoke_rc=$rc3 aggregate_rc=$rc4 end=$(date -u +%FT%TZ)"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ] && [ $rc4 -eq 0 ]
