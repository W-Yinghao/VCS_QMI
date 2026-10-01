#!/usr/bin/env bash
#SBATCH --job-name=p108_estim
#SBATCH --partition=RTX6000PRO,H100,A100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p108_%x_%j.out
# P108 (package v4 module E).  Env: CELLS ('E1:cond:N:seed;E2:cond:orig|rot:N:seed;...'), MECH=1 (E1 mechanism), SMOKE=1, OUT_DIR, CPU=1.
# Resumable: finished cell JSONs are skipped.  CPU smoke: sbatch -p CPU --gres=gpu:0 --export=ALL,SMOKE=1,CPU=1,... slurm/p108_estim.sbatch
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
OUT_DIR=${OUT_DIR:-/home/infres/yinwang/CS_QMI/outputs/P108_estim}
ARGS="--out-dir $OUT_DIR"
[ "${MECH:-0}" = "1" ] && ARGS="$ARGS --mech"
[ "${SMOKE:-0}" = "1" ] && ARGS="$ARGS --smoke"
[ "${CPU:-0}" = "1" ] && ARGS="$ARGS --cpu"
banner "P108 estim: mech=${MECH:-0} smoke=${SMOKE:-0} cells=${CELLS:-none} -> $OUT_DIR"
"$PYTHON" scripts/p108_estim.py $ARGS ${CELLS:+--cells "$CELLS"}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
