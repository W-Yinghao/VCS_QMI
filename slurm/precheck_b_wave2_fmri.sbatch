#!/usr/bin/env bash
#SBATCH --job-name=fmrireg
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/fmrireg_%x_%j.out
# Task pre-check B-T2 (wave 2, third real source): P52 registration protocol on real AOMIC-PIOP1 EPI-boldref / T1w slice pairs. REPORT required;
# DATA = dataset root (default data/aomic_piop1); SMOKE=1 = 3 subjects, 2 inits per radius; EXTRA_ARGS passed through (e.g. --save-slices).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${REPORT:?}"
banner "task pre-check B-T2 fMRI (AOMIC-PIOP1) registration (smoke=${SMOKE:-0})"
if [ "${SMOKE:-0}" = "1" ]; then "$PYTHON" scripts/task_b_fmri.py --data "${DATA:-/home/infres/yinwang/CS_QMI/data/aomic_piop1}" --out "$REPORT" --smoke ${EXTRA_ARGS:-}; rc=$?
else "$PYTHON" scripts/task_b_fmri.py --data "${DATA:-/home/infres/yinwang/CS_QMI/data/aomic_piop1}" --out "$REPORT" ${EXTRA_ARGS:-}; rc=$?; fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
