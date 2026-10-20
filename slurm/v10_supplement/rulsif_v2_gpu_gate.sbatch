#!/usr/bin/env bash
# RuLSIF_exact_v2 CUDA T01/execution gate: thin wrapper over the accepted validate.sbatch
# (CUDA numerics incl. tests/v10/test_rulsif_v2.py, full FIT32 fit, checkpoint resume and
# identity rejection, FD geometry gradient, one complete common search on CAL core 0).
#SBATCH --job-name=v10r2_gate
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --nice=0
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
: "${V10_VALIDATION_SOURCE:?frozen execution worktree required}"
exec bash "$V10_VALIDATION_SOURCE/slurm/v10_supplement/validate.sbatch" gate --family RuLSIF_exact_v2
