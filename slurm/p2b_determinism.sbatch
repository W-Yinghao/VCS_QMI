#!/usr/bin/env bash
#SBATCH --job-name=vcs_p2b_determinism
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p2b_determinism_%j.out
# Diagnostic only (not a gate): two identical continuous 100-step VCS smoke runs on the same GPU, to quantify CUDA run-to-run
# nondeterminism and put the P2 resume difference in context. Also re-compares the P2 interrupted/resumed pair on params vs BN buffers.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P2b determinism diagnostic"
P2JOB=${P2JOB:?set P2JOB to the P2 smoke job id}
for r in a b; do
  run_fg "$PYTHON" -m vcs_ssl.train --config configs/cifar10_pilot_vcs.yaml --smoke-steps 100 --smoke-epoch-steps 50 --run-id "det_vcs_${r}_${SLURM_JOB_ID}" --no-epoch-eval
done
"$PYTHON" scripts/compare_runs.py "$OUTPUT_ROOT/det_vcs_a_${SLURM_JOB_ID}" "$OUTPUT_ROOT/det_vcs_b_${SLURM_JOB_ID}" --title "Two identical continuous runs, same GPU ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))" --report reports/P2b_determinism_identical_runs.md
"$PYTHON" scripts/compare_runs.py "$OUTPUT_ROOT/smoke_vcs_${P2JOB}" "$OUTPUT_ROOT/smoke_vcs_interrupted_${P2JOB}" --title "P2 continuous vs interrupted+resumed (job ${P2JOB})" --report reports/P2b_resume_pair_breakdown.md
echo "end=$(date -u +%FT%TZ)"
