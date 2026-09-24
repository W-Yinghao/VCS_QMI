#!/usr/bin/env bash
#SBATCH --job-name=vcs_p4_summarize
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p4_summarize_%j.out
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P4 summarize (neutral table; failed runs included)"
"$PYTHON" -m vcs_ssl.summarize --stage P3_pilot --output reports/P4_ssl_pilot_results_table.md
rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
