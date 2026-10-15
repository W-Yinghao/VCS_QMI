#!/usr/bin/env bash
#SBATCH --job-name=v9_reg_g2_timing
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --nice=400
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-G2 GPU timing + code-path smokes in a SEPARATE root (outputs_v9/reg_epfl_timing) so nothing here is reused by the formal units:
#   (a) logistic / InfoNCE / CoMIR / kernel smokes on the REG-P3-shaped smoke root (--n-pert 2; code path only);
#   (b) one real unit end to end (train 3 000 steps on the sampled FIT, CAL selection over 12 checkpoints on the 10 CAL windows, DEV 74 windows)
#       with --n-pert 3 (3 of the 10 perturbations per window) to measure training time and s / registration; extrapolated to 1 940 registrations.
# Timing records only — not results of the protocol.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v9; cd "$R" || exit 2; export PYTHONPATH="$R/src:$R"; export V9_REG_EPFL_ROOT=/projects/EEG-foundation-model/yinghao/outputs_v9/reg_epfl_timing
echo "host=$(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git -C "$R" rev-parse --short HEAD) root=$V9_REG_EPFL_ROOT"
rc_all=0
for u in V9-G2-logistic-frac0.25-eps0.2-s901 V9-G2-infonce-frac1-eps0-s901 V9-G2-comir-frac0.25-eps0.2-s901 V9-G2-kernel-frac0.25-eps0.2-s901; do
  t0=$(date +%s); "$PYTHON" scripts/v9/reg_g2.py run --unit "$u" --smoke --n-pert 2; rc=$?; echo "[smoke-exit] $u rc=$rc $(( $(date +%s) - t0 )) s"; [ $rc -ne 0 ] && rc_all=$rc
done
t0=$(date +%s); "$PYTHON" scripts/v9/reg_g2.py run --unit V9-G2-vcs-frac0.25-eps0.2-s901 --n-pert 3; rc=$?; echo "[timing-exit] V9-G2-vcs-frac0.25-eps0.2-s901 n_pert=3 rc=$rc $(( $(date +%s) - t0 )) s"; [ $rc -ne 0 ] && rc_all=$rc
echo "exit=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
