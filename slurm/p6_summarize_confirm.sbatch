#!/usr/bin/env bash
#SBATCH --job-name=vcs_p6_summarize
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p6_summarize_%j.out
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P6 summarize confirm200 (neutral table; failed runs included)"
"$PYTHON" -m vcs_ssl.summarize --stage P5_confirm200 --output reports/P6_confirm200_results_table.md
rc=$?; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
