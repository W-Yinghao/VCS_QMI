#!/usr/bin/env bash
# usage: sbatch --job-name=nr1_<unit> [--mem=..] unit.sbatch <UNIT_ID>
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr1_20261104/sbatch/common_env.sh
U=$1; CFG=${NR1_CFGDIR:-/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/locked_configs}/$U.json
COND=$($NR1_PY -I -c "import json;print(json.load(open('$CFG'))['condition'])")
ARCH=$($NR1_PY -I -c "import json;print(json.load(open('$CFG'))['architecture'])")
RUN=${NR1_RUNS:-$NR1_OUT/runs}/$U; mkdir -p $RUN
echo "$SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)" >> $RUN/ATTEMPTS.txt
if [ "$ARCH" = GL ]; then export NR1_FEATURE_RAM=all; else export NR1_FEATURE_RAM=global; fi
# 1) training process: FIT cache + observed + calibration only (no DEV cache, no truth path)
FITC=${NR1_FIT_CACHE:-$NR1_OUT/cache/fit}; DEVC=${NR1_DEV_CACHE:-$NR1_OUT/cache/dev}
OBS=${NR1_OBS:-$NR1_OUT/data/observed_$COND.tsv}; CAL=${NR1_CAL:-$NR1_OUT/calibration/$COND.json}
TRUTH=${NR1_TRUTH:-$NR1_OUT/evaluator_only/dev_positives.json}
[ -f $RUN/DONE.json ] || $NR1_PY -B site/train_unit.py --config $CFG --cache $FITC \
   --observed $OBS --calibration $CAL --out $RUN
# 2) DEV evaluation: one separate process per registered step; scores written+hashed before truth is read
unset NR1_FEATURE_RAM
for s in 0 250 500 1000 2000 4000 8000; do
  ss=$(printf %06d $s)
  [ -f $RUN/eval_$ss/METRICS.json ] || $NR1_PY -B -m nr1.evaluate --cache $DEVC \
     --truth $TRUTH --checkpoint $RUN/step_$ss.pt --out $RUN/eval_$ss --device cuda --score-dtype float64
done
echo "{\"unit\":\"$U\",\"status\":\"COMPLETE\",\"job\":\"$SLURM_JOB_ID\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $RUN/UNIT_COMPLETE.json
echo "UNIT_COMPLETE $U"
