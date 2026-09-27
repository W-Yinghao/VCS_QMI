#!/usr/bin/env bash
#SBATCH --job-name=precheck_c
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/precheck_c_%x_%j.out
# Pre-check C (batch decoupling) on one GPU. FEATURES and REPORT required; SMOKE=1 for the probe; EXTRA_ARGS passed through.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${FEATURES:?}"; : "${REPORT:?}"
banner "pre-check C batch decoupling (smoke=${SMOKE:-0})"
if [ "${SMOKE:-0}" = "1" ]; then "$PYTHON" scripts/precheck_c_batch.py --features "$FEATURES" --out "$REPORT" --smoke ${EXTRA_ARGS:-}; rc=$?; else "$PYTHON" scripts/precheck_c_batch.py --features "$FEATURES" --out "$REPORT" ${EXTRA_ARGS:-}; rc=$?; fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
