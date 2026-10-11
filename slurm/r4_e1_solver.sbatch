#!/usr/bin/env bash
#SBATCH --job-name=r4_e1
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R4-E1 solver check, one seed per job (scripts/r4_e1_solver.py; args passed through).  16 threads as P152 (CPU arithmetic of the P152 cell).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} start=$(date -u +%FT%TZ)"
"$PYTHON" scripts/r4_e1_solver.py --out $OUTPUT_ROOT/R4_E1 "$@"; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
