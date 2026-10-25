#!/usr/bin/env bash
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nice=60
#SBATCH --no-requeue
#SBATCH --exclude=node51,node53,node60,node61
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE PYTHONPATH
/home/infres/yinwang/CS_QMI/env/bin/python -B -s /home/infres/yinwang/CS_QMI/handoff/r22_a1_complete_20261025/run.py "$@"
if [[ "$1" == "gate" ]]; then /usr/bin/python3 /home/infres/yinwang/CS_QMI/handoff/r22_a1_complete_20261025/admit.py; fi
