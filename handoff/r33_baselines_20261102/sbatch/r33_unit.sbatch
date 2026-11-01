#!/bin/bash
#SBATCH --qos=runfill
#SBATCH --partition=A100,A40,L40S,V100-32GB
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r33_baselines_20261102/logs/%x_%j.out
# Usage: sbatch --job-name=r33_<kind>_<cell> r33_unit.sbatch KIND CELL   (KIND prefix | H0 | a1_iface)
# runfill can preempt (PreemptMode=REQUEUE): the requeued job reruns r33_unit.sh, which resumes from the unit's own state.
set -euo pipefail
echo "=== start $(date -u +%FT%TZ) job ${SLURM_JOB_ID} restart ${SLURM_RESTART_COUNT:-0} host $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
exec bash /home/infres/yinwang/CS_QMI/handoff/r33_baselines_20261102/sbatch/r33_unit.sh "$1" "$2" formal
