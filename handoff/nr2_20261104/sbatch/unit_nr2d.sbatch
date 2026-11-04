#!/usr/bin/env bash
# usage (NR2D): sbatch --job-name=nr2d_<unit> unit_nr2d.sbatch <UNIT_ID>
# Same pipeline as unit.sbatch / unit_nr2c.sbatch; data paths chosen by the config condition.
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr2_20261104/sbatch/common_env.sh
U=$1; CFG=$NR2_H/locked_configs_nr2d/$U.json
$NR1_PY -I -B $NR2_CODE/site/check_frozen_nr2d.py $U
COND=$($NR1_PY -I -c "import json;print(json.load(open('$CFG'))['condition'])")
case $COND in
  COCO00|COCO60|COCO80)
    D=/projects/EEG-foundation-model/yinghao/outputs_nr2/NR2C_20261104
    FITC=$D/cache/fit; DEVC=$D/cache/dev; OBS=$D/data/observed_$COND.tsv; CAL=$D/calibration/$COND.json
    TAB=$D/proposal/base_top100_fit.npz; TRUTH=$D/evaluator_only/dev_positives.json; CLEAN=$D/data/observed_COCO00.tsv
    RUN=$D/runs_nr2d/$U ;;
  CC152K_RECOVERED_20261104)
    D=/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/CC152K_RECOVERED_20261104
    FITC=$D/cache/fit; DEVC=$D/cache/dev; OBS=$D/data/observed_$COND.tsv; CAL=$D/calibration.json
    TAB=/projects/EEG-foundation-model/yinghao/outputs_nr2/NR2D_20261104/proposal_cc/base_top100_fit.npz
    TRUTH=$D/evaluator_only/dev_positives.json; CLEAN=NA
    RUN=/projects/EEG-foundation-model/yinghao/outputs_nr2/NR2D_20261104/runs_cc/$U ;;
  *) echo "unknown condition $COND"; exit 2 ;;
esac
mkdir -p $RUN
echo "$SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)" >> $RUN/ATTEMPTS.txt
export NR1_FEATURE_RAM=global
[ -f $RUN/DONE.json ] || $NR1_PY -B site/train_unit_nr2.py --config $CFG --cache $FITC --observed $OBS --calibration $CAL --tables $TAB --out $RUN
[ -f $RUN/FIT_READOUT.json ] || $NR1_PY -B site/fit_readout.py --run $RUN --out $RUN/FIT_READOUT.json --fit-cache $FITC --observed $OBS --tables $TAB
unset NR1_FEATURE_RAM
for s in 0 250 500 1000 2000 4000 8000; do
  ss=$(printf %06d $s)
  [ -f $RUN/eval_$ss/METRICS.json ] || $NR1_PY -B -m nr1.evaluate --cache $DEVC \
     --truth $TRUTH --checkpoint $RUN/step_$ss.pt --out $RUN/eval_$ss --device cuda --score-dtype float64
done
if [ "$CLEAN" = NA ]; then
  echo '{"evaluator_only": true, "status": "NA", "reason": "real web noise; no injected-noise truth"}' > $RUN/EVALONLY_FIT_STRATA.json
else
  [ -f $RUN/EVALONLY_FIT_STRATA.json ] || $NR1_PY -B site/fit_strata_evalonly.py --run $RUN --out $RUN/EVALONLY_FIT_STRATA.json \
     --fit-cache $FITC --observed $OBS --clean-observed $CLEAN
fi
echo "{\"unit\":\"$U\",\"status\":\"COMPLETE\",\"job\":\"$SLURM_JOB_ID\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $RUN/UNIT_COMPLETE.json
echo "UNIT_COMPLETE $U"
