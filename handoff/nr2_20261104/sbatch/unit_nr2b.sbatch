#!/usr/bin/env bash
# usage (NR2B): sbatch --job-name=nr2b_<unit> unit_nr2b.sbatch <UNIT_ID>
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr2_20261104/sbatch/common_env.sh
U=$1; CFG=$NR2_H/locked_configs_nr2b/$U.json
$NR1_PY -I -B $NR2_CODE/site/check_frozen_nr2b.py $U
PROTO=$($NR1_PY -I -c "import json;print(json.load(open('$CFG'))['protocol_id'])")
COND=$($NR1_PY -I -c "import json;print(json.load(open('$CFG'))['condition'])")
RUN=$NR2_OUT/runs_nr2b/$U; mkdir -p $RUN
echo "$SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)" >> $RUN/ATTEMPTS.txt
FITC=$NR1_OUT/cache/fit; DEVC=$NR1_OUT/cache/dev; OBS=$NR1_OUT/data/observed_$COND.tsv; CAL=$NR1_OUT/calibration/$COND.json
TAB=$NR2_OUT/proposal/base_top100_fit.npz; TRUTH=$NR1_OUT/evaluator_only/dev_positives.json
# 1) training process: FIT cache + observed + calibration (+ truth-free base proposal tables); no DEV cache, no truth path
export NR1_FEATURE_RAM=global
if [ "$PROTO" = NR1_IID_PRODUCT_PQ_V1 ]; then
  [ -f $RUN/DONE.json ] || $NR1_PY -B site/train_unit_nr1.py --config $CFG --cache $FITC --observed $OBS --calibration $CAL --out $RUN
else
  [ -f $RUN/DONE.json ] || $NR1_PY -B site/train_unit_nr2.py --config $CFG --cache $FITC --observed $OBS --calibration $CAL --tables $TAB --out $RUN
fi
# 2) training-side FIT readout (no truth): common product-Q pool + fixed-proposal pools
[ -f $RUN/FIT_READOUT.json ] || $NR1_PY -B site/fit_readout.py --run $RUN --out $RUN/FIT_READOUT.json --fit-cache $FITC --observed $OBS --tables $TAB
unset NR1_FEATURE_RAM
# 3) DEV evaluation (NR1 evaluator, s only): one separate process per step; scores hashed before truth is read
for s in 0 250 500 1000 2000 4000 8000; do
  ss=$(printf %06d $s)
  [ -f $RUN/eval_$ss/METRICS.json ] || $NR1_PY -B -m nr1.evaluate --cache $DEVC \
     --truth $TRUTH --checkpoint $RUN/step_$ss.pt --out $RUN/eval_$ss --device cuda --score-dtype float64
done
# 4) evaluator-only FIT stratification by isolated injected-noise identity (separate process; never used by training)
[ -f $RUN/EVALONLY_FIT_STRATA.json ] || $NR1_PY -B site/fit_strata_evalonly.py --run $RUN --out $RUN/EVALONLY_FIT_STRATA.json \
   --fit-cache $FITC --observed $OBS --clean-observed $NR1_OUT/data/observed_Flickr00.tsv
echo "{\"unit\":\"$U\",\"status\":\"COMPLETE\",\"job\":\"$SLURM_JOB_ID\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $RUN/UNIT_COMPLETE.json
echo "UNIT_COMPLETE $U"
