#!/usr/bin/env bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --nice=60
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE PYTHONPATH
diag_root=/home/infres/yinwang/CS_QMI/handoff/r22_profile_dev_20261023
app=$1; shift
if [ "$app" = reg ]; then
  exec /home/infres/yinwang/CS_QMI/env/bin/python -B -s "$diag_root/registration.py" run "$@"
elif [ "$app" = kd ]; then
  exec /home/infres/yinwang/CS_QMI/env_kd/bin/python -B -s "$diag_root/kd_probe.py" run "$@"
else
  exit 2
fi
