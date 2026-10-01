#!/usr/bin/env bash
#SBATCH --job-name=p105_t1_ablation
#SBATCH --partition=RTX6000PRO,H100,A100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p105_%x_%j.out
# P105 T1 critic-class ablation.  Env: FEATURES (P45 dir), FAMILY (colour|blur), CELLS ('case:n;case:n'), SEED, REPORT (prefix), REPEATS (default 100),
# SMOKE=1 -> 3 repeats, 20 permutations, 60 steps, n = 200.  CPU smoke: sbatch -p CPU --gres=gpu:0 ...
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${FEATURES:?}" "${FAMILY:?}" "${CELLS:?}" "${SEED:?}" "${REPORT:?}"
EXTRA=""; [ "${SMOKE:-0}" = "1" ] && EXTRA="--smoke"
banner "P105 T1 ablation: $FEATURES $FAMILY cells=$CELLS seed=$SEED repeats=${REPEATS:-100} -> $REPORT smoke=${SMOKE:-0}"
"$PYTHON" scripts/cond_test_t1_ablation.py --features "$FEATURES" --family "$FAMILY" --cells "$CELLS" --seed "$SEED" --repeats "${REPEATS:-100}" --out "$REPORT" $EXTRA; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
