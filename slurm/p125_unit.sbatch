#!/usr/bin/env bash
#SBATCH --job-name=p125_unit
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P125 runner job.  Env: STAGE (pilot|confirm) or CELLS ('cond:N:seed:orig|rot;...'), OUT_DIR, CPU=1 forces CPU even on a GPU node.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
OUT_DIR=${OUT_DIR:-/home/infres/yinwang/CS_QMI/outputs/P125_struct}; ARGS="--out-dir $OUT_DIR"
[ -n "${STAGE:-}" ] && ARGS="$ARGS --stage $STAGE"; [ "${CPU:-0}" = "1" ] && ARGS="$ARGS --cpu"
banner "P125 struct: stage=${STAGE:-} cells=${CELLS:-} -> $OUT_DIR"
"$PYTHON" scripts/p125_struct.py $ARGS ${CELLS:+--cells "$CELLS"}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
