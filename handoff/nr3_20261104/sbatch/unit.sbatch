#!/usr/bin/env bash
# usage: sbatch --job-name=nr3_<unit> unit.sbatch <UNIT_ID>
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr3_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr3_20261104/sbatch/common_env.sh
U=$1; CFG=$NR3_H/configs_production/$U.json; PK=$NR3_H/package/configs/$U.json
$NR1_PY -I -B $NR3_H/code/nr3/check_frozen_nr3.py $U
C=$($NR1_PY -I -c "import json;print(json.load(open('$PK'))['condition'])"); BLK=$($NR1_PY -I -c "import json;print(json.load(open('$PK'))['block'])")
pth() { $NR1_PY -I -c "import sys;sys.path.insert(0,'$NR3_H/code');from nr3.paths import cond_paths;print(cond_paths('$C')['$1'])"; }
FITC=$(pth fit); DEVC=$(pth dev); OBS=$(pth observed); CAL=$(pth calibration); TAB=$(pth tables); TRUTH=$(pth truth)
OD=$NR3_OUT/prep/$C; RUN=$NR3_OUT/runs/$U; mkdir -p $RUN
echo "$SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)" >> $RUN/ATTEMPTS.txt
export NR1_FEATURE_RAM=global
if [ "$BLK" = A_HREF_CALIBRATED_INIT ]; then
  [ -f $RUN/DONE.json ] || $NR1_PY -B $NR2_CODE/site/train_unit_nr2.py --config $CFG --cache $FITC --observed $OBS --calibration $OD/calibration_HCAL.json --tables $TAB --out $RUN
  STEPS="0 250 500 1000 2000 4000 8000"
else
  [ -f $RUN/DONE.json ] || $NR1_PY -B -m nr3.train_unit_b --config $CFG --cache $FITC --observed $OBS --calibration $CAL --prefix $OD/prefix/PREFIX_step_008000.pt --tables $TAB --out $RUN
  STEPS="8000 8250 8500 9000 10000 12000"
fi
[ -f $RUN/NR3_READOUT.json ] || $NR1_PY -B -m nr3.readout --run $RUN --cond $C
unset NR1_FEATURE_RAM
for s in $STEPS; do ss=$(printf %06d $s)
  [ -f $RUN/eval_$ss/METRICS.json ] || $NR1_PY -B -m nr1.evaluate --cache $DEVC --truth $TRUTH --checkpoint $RUN/step_$ss.pt --out $RUN/eval_$ss --device cuda --score-dtype float64
done
echo "{\"unit\":\"$U\",\"status\":\"COMPLETE\",\"job\":\"$SLURM_JOB_ID\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $RUN/UNIT_COMPLETE.json
echo "UNIT_COMPLETE $U"
