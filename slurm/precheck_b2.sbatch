#!/usr/bin/env bash
#SBATCH --job-name=precheck_b2
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/precheck_b2_%x_%j.out
# Pre-check B2 (registration energy surfaces) on one GPU. REPORT required; SMOKE=1 for the probe.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${REPORT:?}"
banner "pre-check B2 registration (smoke=${SMOKE:-0})"
if [ "${SMOKE:-0}" = "1" ]; then "$PYTHON" scripts/precheck_b2_registration.py --out "$REPORT" --smoke; rc=$?; else "$PYTHON" scripts/precheck_b2_registration.py --out "$REPORT" ${EXTRA_ARGS:-}; rc=$?; fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
