#!/usr/bin/env bash
#SBATCH --job-name=precheck_a2
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/precheck_a_wave2_%x_%j.out
# Pre-check A wave 2 (A-S1 curve + A-T task) on one GPU. SHIFT_DIRS (name=dir[,name=dir]; pass through the environment, not --export) and REPORT required;
# SMOKE=1 for the CPU probe (submit with -p CPU --gres=gpu:0); PAIRINGS / EXTRA_ARGS optional.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${SHIFT_DIRS:?}"; : "${REPORT:?}"
banner "pre-check A wave 2 (smoke=${SMOKE:-0}) <- ${SHIFT_DIRS}"
ARGS="--shift-dirs $SHIFT_DIRS --out $REPORT --pairings ${PAIRINGS:-exact,topic,coarse,random} ${EXTRA_ARGS:-}"
if [ "${SMOKE:-0}" = "1" ]; then ARGS="$ARGS --smoke"; fi
"$PYTHON" scripts/precheck_a_wave2.py run $ARGS; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
