#!/usr/bin/env bash
#SBATCH --job-name=p139_loader
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p139_loader_%j.out
# P139 CPU stage throughput: decode once + per-view RandomResizedCrop + pil_to_tensor (uint8), 2 / 4 views, 4-32 workers, disjoint subsets.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
set -o pipefail
banner "P139 loader (CPU stage of the GPU-augmentation pipeline)"
"$PYTHON" scripts/p139_imagenet_pipeline.py loader --out reports/P139/loader_${SLURM_JOB_ID}.json 2>&1 | tail -12; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
