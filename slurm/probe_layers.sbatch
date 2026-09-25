#!/usr/bin/env bash
#SBATCH --job-name=probe_layers
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/probe_layers_%j.out
# Read-only per-layer probe diagnostic on frozen checkpoints (CPU partition, so it does not take a GPU from the training units). RUNS colon-separated.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
banner "per-layer probe diagnostic"
"$PYTHON" scripts/probe_layers.py --runs "${RUNS//:/,}" --out "${OUT:-reports/PROBE_LAYERS.md}" --num-workers 4
rc=$?; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
