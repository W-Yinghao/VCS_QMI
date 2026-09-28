#!/usr/bin/env bash
#SBATCH --job-name=robust_r2r3
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/robust_r2r3_%x_%j.out
# R line (P71): R2 noisy pairing + R3 mismatch identification on one GPU. SHIFT_DIRS (name=dir[,name=dir]; pass through the environment, not
# --export) and REPORT required; SMOKE=1 for the CPU probe (submit with -p CPU --gres=gpu:0); EXTRA_ARGS optional.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${SHIFT_DIRS:?}"; : "${REPORT:?}"
banner "R2/R3 (smoke=${SMOKE:-0}) <- ${SHIFT_DIRS}"
ARGS="--shift-dirs $SHIFT_DIRS --out $REPORT ${EXTRA_ARGS:-}"
if [ "${SMOKE:-0}" = "1" ]; then ARGS="$ARGS --smoke"; fi
"$PYTHON" scripts/robust_r2r3.py $ARGS; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
