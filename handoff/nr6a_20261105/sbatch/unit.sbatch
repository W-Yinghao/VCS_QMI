#!/usr/bin/env bash
# NR6-A unit (NR5_A1 trainer/evaluator, one GPU type per matched group). usage: sbatch --partition=<type> --mem=.. --time=.. --job-name=nr6a_<unit> unit.sbatch <UNIT_ID>
#SBATCH --qos=normal
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr6a_20261105/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr6a_20261105/sbatch/common_env.sh
U=$1; CFG=$NR6_H/configs_production/$U.json
$NR1_PY -I -B $NR6_H/code/nr6a/check_frozen_nr6a.py $U
jq() { $NR1_PY -I -c "import json;v=json.load(open('$CFG')).get('$1');print('' if v is None else v)"; }
C=$(jq condition); SC=$(jq scope); INIT=$(jq init); PRE=$(jq prefix);
pth() { $NR1_PY -I -c "import sys;sys.path[:0]=['$NR3_CODE','$NR4_H/code','$NR5_H/code'];from nr5.paths import paths;print(paths('$C')['$1'])"; }
FITC=$(pth fit); DEVC=$(pth dev); OBS=$(pth observed); CAL=$(pth calibration); TAB=$(pth tables); TRUTH=$(pth truth); HID=$(pth hidden)
RUN=$NR6_OUT/runs/$U; mkdir -p $RUN
echo "$SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)" >> $RUN/ATTEMPTS.txt
export NR1_FEATURE_RAM=global NR4_HIDDEN_RAM=all; [ "$C" = COCO60 ] && export NR4_HIDDEN_RAM=img
EXTRA=""; [ "$SC" = TAIL_LORA8 ] && EXTRA="--hidden $HID"
OBJ=$(jq objective)
if [ "$INIT" = NATIVE ]; then P0=0; S0ARG="--stage0-base"; else P0=8000; EXTRA="$EXTRA --prefix $PRE"; S0ARG="--ref-stage0 /projects/EEG-foundation-model/yinghao/outputs_nr3/NR3_20261104/runs/NR3_B_NCE2${OBJ}_${C}_G_s1001/eval_008000/PREDICTIONS.npz"; fi
[ -f $RUN/DONE.json ] || $NR1_PY -B -m nr5a1.train_unit --config $CFG --cache $FITC --observed $OBS --calibration $CAL --tables $TAB --out $RUN $EXTRA
unset NR1_FEATURE_RAM
LAST=$($NR1_PY -I -c "import json;print(max(json.load(open('$CFG'))['score_checks']))")
for s in $($NR1_PY -I -c "import json;print(' '.join(map(str,json.load(open('$CFG'))['score_checks'])))"); do ss=$(printf %06d $((P0+s)))
  FG=""; [ "$s" = "$LAST" ] && FG="--full-gallery"
  [ -f $RUN/eval_$ss/METRICS.json ] || $NR1_PY -B -m nr5a1.evaluate --cond $C --cache $DEVC --truth $TRUTH --checkpoint $RUN/step_$ss.pt --out $RUN/eval_$ss $S0ARG $FG
done
echo "{\"unit\":\"$U\",\"status\":\"COMPLETE\",\"job\":\"$SLURM_JOB_ID\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $RUN/UNIT_COMPLETE.json
echo "UNIT_COMPLETE $U"
