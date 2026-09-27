#!/usr/bin/env bash
#SBATCH --job-name=ixireg
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/ixireg_%x_%j.out
# Task pre-check B-T2 (wave 2): P52 registration protocol on real IXI PD/T2 slice pairs. REPORT required; DATA = dataset root (default data/ixi);
# SMOKE=1 = 3 real subjects, 2 inits; SYNTH=<dir> = code probe on constructed NIfTI pairs (no dataset needed).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${REPORT:?}"
banner "task pre-check B-T2 IXI registration (smoke=${SMOKE:-0} synth=${SYNTH:-})"
if [ -n "${SYNTH:-}" ]; then "$PYTHON" scripts/task_b_ixi.py --synthetic-smoke "$SYNTH" --out "$REPORT"; rc=$?
elif [ "${SMOKE:-0}" = "1" ]; then "$PYTHON" scripts/task_b_ixi.py --data "${DATA:-/home/infres/yinwang/CS_QMI/data/ixi}" --out "$REPORT" --smoke; rc=$?
else "$PYTHON" scripts/task_b_ixi.py --data "${DATA:-/home/infres/yinwang/CS_QMI/data/ixi}" --out "$REPORT" ${EXTRA_ARGS:-}; rc=$?; fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
