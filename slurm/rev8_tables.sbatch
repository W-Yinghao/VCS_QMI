#!/bin/bash
#SBATCH --job-name=rev8_tables
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
cd /home/infres/yinwang/CS_QMI/ssl_pilot
/home/infres/yinwang/CS_QMI/env/bin/python -I scripts/rev8_tables.py reports/REV8/REV8_DIAG_RESULTS/rev8_diag_results.json > "$1"
echo "exit=0 end=$(date -u +%FT%TZ)"
