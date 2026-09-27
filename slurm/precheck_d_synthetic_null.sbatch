#!/usr/bin/env bash
#SBATCH --job-name=precheck_d_synnull
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/precheck_d_synnull_%x_%j.out
# Pre-check D addendum 6: synthetic null level check on one GPU. MODE (fresh|fixed|fixed_redraw) and REPORT (prefix) required; ARGS passed through
# (pass comma lists through the environment, not --export).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${MODE:?}"; : "${REPORT:?}"
banner "pre-check D synthetic null (${MODE}) -> ${REPORT}"
"$PYTHON" scripts/precheck_d_synthetic_null.py --mode "$MODE" --out "$REPORT" ${ARGS:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
