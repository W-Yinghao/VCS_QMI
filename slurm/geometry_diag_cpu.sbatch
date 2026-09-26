#!/usr/bin/env bash
#SBATCH --job-name=geom_diag_cpu
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/geom_diag_cpu_%j.out
# Read-only geometry diagnostic on the CPU partition (does not take a GPU from training). RUNS colon-separated; OUT = report path.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
banner "geometry diagnostic (CPU)"
"$PYTHON" scripts/alignment_uniformity.py --runs "${RUNS//:/,}" --out "${OUT:-reports/GEOMETRY_DIAG.md}"
rc=$?; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
