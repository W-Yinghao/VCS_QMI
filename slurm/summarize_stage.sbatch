#!/usr/bin/env bash
#SBATCH --job-name=vcs_summarize
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/summarize_%x_%j.out
# sbatch --job-name=p9_summarize --export=ALL,STAGE=P8_long800,OUT=reports/P9_long800_results_table.md summarize_stage.sbatch
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${STAGE:?}"; : "${OUT:?}"
banner "summarize ${STAGE} -> ${OUT}"
"$PYTHON" -m vcs_ssl.summarize --stage "$STAGE" --output "$OUT"; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
