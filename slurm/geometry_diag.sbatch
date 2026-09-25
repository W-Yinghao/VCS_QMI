#!/usr/bin/env bash
#SBATCH --job-name=geom_diag
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/geom_diag_%j.out
# Read-only geometry diagnostic (alignment/uniformity/cosine quantiles) on frozen checkpoints. RUNS colon-separated.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "geometry diagnostic"
"$PYTHON" scripts/alignment_uniformity.py --runs "${RUNS//:/,}" --out reports/GEOMETRY_DIAG.md
rc=$?; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
