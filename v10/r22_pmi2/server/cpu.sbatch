#!/usr/bin/env bash
#SBATCH --qos=normal
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R22-C2 CPU check (check_cpu.py): selection reproduction on the historical J CAL rows, PMI path = R22-C, real-stack dry run on a
# reduced CPU grid (2 CAL + 2 DEV problems; no TEST).  Writes v10/r22_pmi2/results/CPU_CHECK.json.
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE PYTHONPATH
WT=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_r22c
echo "host=$(hostname) start=$(date -u +%FT%TZ) job=${SLURM_JOB_ID} commit=$(git -C "$WT" rev-parse HEAD)"
if [ -n "$(git -C "$WT" status --porcelain -- src v10/r22_pmi/pmi_energy.py v10/r22_pmi/run_unit.py v10/r22_pmi2/run_pmi2.py v10/r22_pmi2/check_cpu.py v10/r22_pmi2/units v10/epfl/r22_pmi2/units)" ]; then echo "REFUSED: uncommitted code changes"; exit 3; fi
/home/infres/yinwang/CS_QMI/env/bin/python -B -s "$WT/v10/r22_pmi2/check_cpu.py"
echo "end=$(date -u +%FT%TZ)"
