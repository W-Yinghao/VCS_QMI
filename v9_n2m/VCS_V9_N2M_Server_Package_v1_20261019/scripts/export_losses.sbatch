#!/usr/bin/env bash
#SBATCH --job-name=v9n2m_lossdiag
#SBATCH --partition=RTX6000PRO
#SBATCH --qos=normal
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
if [ "$#" -ne 5 ]; then
    echo 'usage: sbatch export_losses.sbatch PACKAGE_ROOT EXPERIMENT_JSON SEED OUT_CSV PYTHON_EXECUTABLE' >&2
    exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
exec "$5" -u "$1/run_n2m.py" export-fixed-losses --experiment "$2" --seed "$3" --out-csv "$4"
