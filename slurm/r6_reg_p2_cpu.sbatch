#!/usr/bin/env bash
#SBATCH --job-name=r6_reg_p2_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R6 REG-P2 CPU steps (scripts/r6_reg_p2.py perturb | register ... [--smoke]); args passed through.  Main env; threads = allocated CPUs.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-NA} start=$(date -u +%FT%TZ)"
"$PYTHON" scripts/r6_reg_p2.py "$@"; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
