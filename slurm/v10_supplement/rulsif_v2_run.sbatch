#!/usr/bin/env bash
# RuLSIF_exact_v2 formal CAL / final-seed job: thin wrapper over the accepted run.sbatch
# (allocation from Slurm StartTime, exit receipt, signal forwarding). Arguments:
#   cal RuLSIF_exact_v2            9 pre-registered candidates in order, seed 1017, 8 GPU-h cap
#   final RuLSIF_exact_v2 --seed S S in 1001 1002 1003; DEV32 at the CAL-selected candidate
#SBATCH --job-name=v10r2_run
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --nice=0
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
: "${V10_SUPPLEMENT_SOURCE:?frozen execution worktree required}"
exec bash "$V10_SUPPLEMENT_SOURCE/slurm/v10_supplement/run.sbatch" "$@"
