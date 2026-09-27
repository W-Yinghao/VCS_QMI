#!/usr/bin/env bash
#SBATCH --job-name=rgbnir
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/rgbnir_%x_%j.out
# Task pre-check B-T2 (wave 2): P52 registration protocol on real RGB-NIR pairs. REPORT required; DATA = dataset root (default nirscene1);
# SMOKE=1 = 3 real pairs; SYNTH=<dir> = code probe on constructed pairs in the dataset layout (no dataset needed).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${REPORT:?}"
banner "task pre-check B-T2 RGB-NIR registration (smoke=${SMOKE:-0} synth=${SYNTH:-})"
if [ -n "${SYNTH:-}" ]; then "$PYTHON" scripts/task_b_rgbnir.py --synthetic-smoke "$SYNTH" --out "$REPORT"; rc=$?
elif [ "${SMOKE:-0}" = "1" ]; then "$PYTHON" scripts/task_b_rgbnir.py --data "${DATA:-/home/infres/yinwang/CS_QMI/data/rgb_nir/nirscene1}" --out "$REPORT" --smoke; rc=$?
else "$PYTHON" scripts/task_b_rgbnir.py --data "${DATA:-/home/infres/yinwang/CS_QMI/data/rgb_nir/nirscene1}" --out "$REPORT" ${EXTRA_ARGS:-}; rc=$?; fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
