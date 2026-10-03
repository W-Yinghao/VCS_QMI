#!/usr/bin/env bash
#SBATCH --job-name=p117_cal
#SBATCH --partition=A100,L40S,RTX6000PRO,H100
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p117_%x_%j.out
# P117 (v5 NEXT-E-CAL).  Env: CELLS ('cond:N:seed;...'), SMOKE=1, OUT_DIR, CPU=1.  Resumable (finished cell JSONs are skipped).
# CPU gate / smoke: sbatch -p CPU --gres=gpu:0 --export=ALL,SMOKE=1,CPU=1,CELLS=... slurm/p117_cal.sbatch
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
OUT_DIR=${OUT_DIR:-/home/infres/yinwang/CS_QMI/outputs/P117_cal}
ARGS="--out-dir $OUT_DIR"
[ "${SMOKE:-0}" = "1" ] && ARGS="$ARGS --smoke"
[ "${CPU:-0}" = "1" ] && ARGS="$ARGS --cpu"
banner "P117 calibration: smoke=${SMOKE:-0} cells=${CELLS:-none} -> $OUT_DIR"
"$PYTHON" scripts/p117_cal.py $ARGS --cells "${CELLS:?}"; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
