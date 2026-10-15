#!/usr/bin/env bash
#SBATCH --job-name=p157_aggregate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p157_aggregate_%j.out
# P157 (SEP-2) aggregate after the population job (protocol reports/P157_SEP2_PROTOCOL_FROZEN_20261015.md §Aggregation).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; cd /home/infres/yinwang/CS_QMI/ssl_pilot
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) cells=$(ls /home/infres/yinwang/CS_QMI/outputs/P157_sep/P156_*_p157.json 2>/dev/null | wc -l)"
mkdir -p reports/P157_SEP2_results
"$PYTHON" scripts/p157_aggregate.py --dir /home/infres/yinwang/CS_QMI/outputs/P157_sep --out reports/P157_SEP2_results/p157_aggregate; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
