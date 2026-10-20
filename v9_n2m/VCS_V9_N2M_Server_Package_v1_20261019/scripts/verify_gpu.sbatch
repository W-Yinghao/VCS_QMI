#!/usr/bin/env bash
#SBATCH --job-name=v9n2m_verify
#SBATCH --partition=RTX6000PRO
#SBATCH --qos=normal
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
if [ "$#" -ne 4 ]; then
    echo 'usage: sbatch verify_gpu.sbatch PACKAGE_ROOT BASE_HANDOFF NEW_OUTPUT PYTHON_EXECUTABLE' >&2
    exit 2
fi
N2M_PACKAGE="$1"
N2M_BASE="$2"
N2M_OUT="$3"
N2M_PYTHON="$4"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
exec "$N2M_PYTHON" -u "$N2M_PACKAGE/run_n2m.py" verify-torch --base "$N2M_BASE" --out "$N2M_OUT" --device cuda
