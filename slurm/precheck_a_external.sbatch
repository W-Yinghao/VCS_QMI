#!/usr/bin/env bash
#SBATCH --job-name=precheck_a_ext
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/precheck_a_ext_%x_%j.out
# Pre-check A addendum 1 (external target). SHIFT_DIRS (name=dir[,name=dir]; via the environment, not --export), EXT (features dir), REPORT required;
# SMOKE=1 for the CPU probe (submit with -p CPU --gres=gpu:0); EXTRA_ARGS optional.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${SHIFT_DIRS:?}"; : "${EXT:?}"; : "${REPORT:?}"
banner "pre-check A external target (smoke=${SMOKE:-0}) <- ${SHIFT_DIRS} ; ext ${EXT}"
ARGS="--shift-dirs $SHIFT_DIRS --ext $EXT --out $REPORT ${EXTRA_ARGS:-}"; [ "${SMOKE:-0}" = "1" ] && ARGS="$ARGS --smoke"
"$PYTHON" scripts/precheck_a_external.py run $ARGS; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
