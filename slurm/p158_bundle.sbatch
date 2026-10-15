#!/usr/bin/env bash
#SBATCH --job-name=p158_bundle
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P158 full run in ONE job (sequential units: null, weak, moderate; finished condition files are skipped), then the aggregate.
# Normal QOS only.  Override: CONDS (default null,weak,moderate), REPEATS (5), OUT (reports/P158/truth).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
OUT=${OUT:-reports/P158/truth}; mkdir -p "$(dirname $OUT)"; banner "P158 bundle conds=${CONDS:-null,weak,moderate} repeats=${REPEATS:-5}"
echo "commit=$(git rev-parse --short HEAD) code=$(cat scripts/p158_common.py scripts/p158_truth.py src/vcs_measure/nested_joint.py | sha256sum | cut -c1-16)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
rc_all=0
for c in ${CONDS//,/ }; do
  "$PYTHON" scripts/p158_truth.py --conds $c --repeats ${REPEATS:-5} --out $OUT; rc=$?; echo "[unit-exit] $c rc=$rc $(date -u +%FT%TZ)"; [ $rc -ne 0 ] && rc_all=$rc
done
"$PYTHON" scripts/p158_aggregate.py --in $OUT --out "$(dirname $OUT)/p158_results"; rc=$?; [ $rc -ne 0 ] && rc_all=$rc
echo "exit=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
