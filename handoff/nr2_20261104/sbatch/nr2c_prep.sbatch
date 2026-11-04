#!/usr/bin/env bash
# NR2C prep: global-only SigLIP2 export (dev, fit), FIT-only calibration per condition, base DEV readout,
# truth-free FIT Top100 proposal tables, then COCO acceptance. TEST is never extracted or read.
#SBATCH --job-name=nr2c_prep
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr2_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr2_20261104/sbatch/common_env.sh
OC=/projects/EEG-foundation-model/yinghao/outputs_nr2/NR2C_20261104; REV=75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
for split in dev fit; do
  if [ ! -f $OC/cache/$split/manifest.json ]; then s=$SECONDS
    $NR1_PY -B -m nr2.extract_global --images $OC/data/${split}_images.jsonl --texts $OC/data/${split}_texts.jsonl \
      --split $split --out $OC/cache/$split --revision $REV --device cuda --batch-size 32 --local-only
    echo "{\"split\":\"$split\",\"seconds\":$((SECONDS-s)),\"job\":\"$SLURM_JOB_ID\",\"node\":\"$(hostname)\",\"gpu\":\"$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)\"}" > $OC/cache/$split/EXTRACT_COST.json
  fi
done
mkdir -p $OC/calibration $OC/baseline
for c in COCO00 COCO60 COCO80; do
  [ -f $OC/calibration/$c.json ] || NR1_FEATURE_RAM=global $NR1_PY -B -m nr1.calibrate --cache $OC/cache/fit --observed $OC/data/observed_$c.tsv --out $OC/calibration/$c.json
  cat $OC/calibration/$c.json
done
[ -f $OC/baseline/float64/METRICS.json ] || $NR1_PY -B -m nr1.evaluate --cache $OC/cache/dev --truth $OC/evaluator_only/dev_positives.json --out $OC/baseline/float64 --device cuda --score-dtype float64
[ -f $OC/proposal/base_top100_fit.npz ] || $NR1_PY -B site/build_tables_nr2c.py --cache $OC/cache/fit --out $OC/proposal --margin 400
export NR1_FEATURE_RAM=global
$NR1_PY -B site/nr2c_acceptance.py
echo NR2C_PREP_DONE
