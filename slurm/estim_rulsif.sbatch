#!/usr/bin/env bash
#SBATCH --job-name=estim_rulsif
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/estim_rulsif_%j.out
# E4: closed-form VCS vs RuLSIF. REPORT prefix required; EXTRA_ARGS passed through.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${REPORT:?}"
banner "E4 RuLSIF vs closed-form VCS -> $REPORT (${EXTRA_ARGS:-})"
"$PYTHON" scripts/estim_rulsif.py --out "$REPORT" ${EXTRA_ARGS:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
