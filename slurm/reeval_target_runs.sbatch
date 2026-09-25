#!/usr/bin/env bash
#SBATCH --job-name=reeval_target
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/reeval_target_%j.out
# Re-run the frozen evaluation (features cached) for completed target-branch / predictor runs so heldout-J follows the training wiring.
# RUNS is colon-separated (sbatch --export splits on commas).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "re-evaluate target-branch/predictor runs with corrected hold-out wiring"
for r in ${RUNS//:/ }; do
  d="$OUTPUT_ROOT/$r"; [ -f "$d/evaluations/evaluation_epoch_200.json" ] || { echo "[skip] $r (no final eval yet)"; continue; }
  cp "$d/evaluations/evaluation_epoch_200.json" "$d/evaluations/evaluation_epoch_200.pre_wiring_fix.json"
  "$PYTHON" -m vcs_ssl.evaluate --run-dir "$d" --checkpoint epoch_200.pt --protocol pilot || echo "[fail] $r"
done
echo "end=$(date -u +%FT%TZ)"
