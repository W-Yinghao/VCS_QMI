#!/usr/bin/env bash
# usage: sbatch [--partition/--mem per scope] --job-name=nr4_<unit> unit.sbatch <UNIT_ID>
#SBATCH --qos=normal
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr4_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr4_20261104/sbatch/common_env.sh
U=$1; CFG=$NR4_H/configs_production/$U.json
$NR1_PY -I -B $NR4_H/code/nr4/check_frozen_nr4.py $U
jq() { $NR1_PY -I -c "import json;print(json.load(open('$CFG'))['$1'])"; }
C=$(jq condition); SC=$(jq scope); REF=$(jq reference_unit)
pth() { $NR1_PY -I -c "import sys;sys.path[:0]=['$NR3_CODE','$NR4_H/code'];from nr4.paths import paths;print(paths('$C','$REF')['$1'])"; }
FITC=$(pth fit); DEVC=$(pth dev); OBS=$(pth observed); CAL=$(pth calibration); TAB=$(pth tables); TRUTH=$(pth truth); PRE=$(pth prefix); HID=$(pth hidden); NR3R=$(pth nr3_run)
RUN=$NR4_OUT/runs/$U; mkdir -p $RUN
echo "$SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)" >> $RUN/ATTEMPTS.txt
export NR1_FEATURE_RAM=global
case $C in COCO60) export NR4_HIDDEN_RAM=img;; *) export NR4_HIDDEN_RAM=all;; esac
EXTRA=""; [ "$SC" = TAIL_LORA8 ] && EXTRA="--hidden $HID"
[ -f $RUN/DONE.json ] || $NR1_PY -B -m nr4.train_unit --config $CFG --cache $FITC --observed $OBS --calibration $CAL --prefix $PRE --tables $TAB --out $RUN $EXTRA
unset NR1_FEATURE_RAM
for s in 8000 8250 8500 9000 10000 12000; do ss=$(printf %06d $s)
  [ -f $RUN/eval_$ss/METRICS.json ] || $NR1_PY -B -m nr4.evaluate --cond $C --cache $DEVC --truth $TRUTH --checkpoint $RUN/step_$ss.pt --out $RUN/eval_$ss --ref-stage0 $NR3R/eval_008000/PREDICTIONS.npz
done
echo "{\"unit\":\"$U\",\"status\":\"COMPLETE\",\"job\":\"$SLURM_JOB_ID\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $RUN/UNIT_COMPLETE.json
echo "UNIT_COMPLETE $U"
