#!/usr/bin/env bash
#SBATCH --job-name=b2var
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/b2var_%x_%j.out
# Pre-check B-S2: registration-energy variants (wave 2). REPORT (prefix) required; EXTRA_ARGS = variant flags (pass via the environment, not --export);
# SMOKE=1 runs the 3-image probe.  Same 60 val2017 images and seed as P52.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${REPORT:?}"
banner "pre-check B-S2 registration variant: ${EXTRA_ARGS:-<none>} (smoke=${SMOKE:-0})"
if [ "${SMOKE:-0}" = "1" ]; then "$PYTHON" scripts/precheck_b2_registration.py --out "$REPORT" --smoke ${EXTRA_ARGS:-}; rc=$?; else "$PYTHON" scripts/precheck_b2_registration.py --out "$REPORT" ${EXTRA_ARGS:-}; rc=$?; fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
