#!/usr/bin/env bash
#SBATCH --job-name=precheck_a_adapt
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/precheck_a_adapt_%x_%j.out
# Pre-check A step 2 (adapters, calibration, threshold transfer) on one GPU. FEATURES and REPORT required; SMOKE=1 for the probe.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${FEATURES:?}"; : "${REPORT:?}"
banner "pre-check A adapters (smoke=${SMOKE:-0}) <- ${FEATURES}"
if [ "${SMOKE:-0}" = "1" ]; then "$PYTHON" scripts/precheck_a_adapters.py --features "$FEATURES" --out "$REPORT" --smoke; rc=$?; else "$PYTHON" scripts/precheck_a_adapters.py --features "$FEATURES" --out "$REPORT" ${EXTRA_ARGS:-}; rc=$?; fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
