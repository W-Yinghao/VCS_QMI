#!/bin/bash
#SBATCH --qos=runfill
#SBATCH --partition=V100-32GB,L40S,A100,A40
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r31_in2r_fidelity_20261101/logs/%x_%j.out
# Usage: sbatch --job-name=r31_<cell>_<arm> r31_unit.sbatch CELL ARM
# runfill can preempt (PreemptMode=REQUEUE): the requeued job reruns this script and resumes from state_last.pt.
set -euo pipefail
echo "=== start $(date -u +%FT%TZ) job ${SLURM_JOB_ID} restart ${SLURM_RESTART_COUNT:-0} host $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
exec bash /home/infres/yinwang/CS_QMI/handoff/r31_in2r_fidelity_20261101/sbatch/r31_unit.sh "$1" "$2" formal
