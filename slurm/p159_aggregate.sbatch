#!/usr/bin/env bash
#SBATCH --job-name=p159_aggregate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p159_aggregate_%j.out
# P159 (SEP-2) aggregate after the population job (protocol reports/P159_SEP3_NATURAL_PREREG_FROZEN_20261015.md §Cost and submission).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; cd /home/infres/yinwang/CS_QMI/ssl_pilot
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) cells=$(ls /home/infres/yinwang/CS_QMI/outputs/P159_sep3/P159_S3_*.json 2>/dev/null | wc -l)"
mkdir -p reports/P159_SEP3_results
"$PYTHON" scripts/p159_aggregate.py --dir /home/infres/yinwang/CS_QMI/outputs/P159_sep3 --out reports/P159_SEP3_results/p159_aggregate; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
