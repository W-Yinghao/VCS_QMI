#!/usr/bin/env bash
#SBATCH --job-name=nr1_extract
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr1_20261104/sbatch/common_env.sh
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
T() { date -u +%FT%TZ; }
for split in dev fit; do
  bs=32  # fixed forward shape, identical to acceptance (amendment A1)
  if [ ! -f $NR1_OUT/cache/$split/manifest.json ]; then
    echo "$(T) extract $split start"; s=$SECONDS
    $NR1_PY -B -m nr1.extract_siglip2 --images $NR1_OUT/data/${split}_images.jsonl --texts $NR1_OUT/data/${split}_texts.jsonl \
      --split $split --out $NR1_OUT/cache/$split --revision $NR1_REV --device cuda --batch-size $bs --local-only
    echo "{\"split\":\"$split\",\"seconds\":$((SECONDS-s)),\"job\":\"$SLURM_JOB_ID\",\"node\":\"$(hostname)\",\"gpu\":\"$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)\"}" > $NR1_OUT/cache/$split/EXTRACT_COST.json
    echo "$(T) extract $split done $((SECONDS-s))s"
  fi
done
du -sb $NR1_OUT/cache/*
mkdir -p $NR1_OUT/calibration
for c in Flickr00 Flickr60 Flickr80; do
  [ -f $NR1_OUT/calibration/$c.json ] || NR1_FEATURE_RAM=global $NR1_PY -B -m nr1.calibrate --cache $NR1_OUT/cache/fit --observed $NR1_OUT/data/observed_$c.tsv --out $NR1_OUT/calibration/$c.json
  cat $NR1_OUT/calibration/$c.json
done
mkdir -p $NR1_OUT/baseline
for dt in float64 float32; do
  [ -f $NR1_OUT/baseline/$dt/METRICS.json ] || $NR1_PY -B -m nr1.evaluate --cache $NR1_OUT/cache/dev --truth $NR1_OUT/evaluator_only/dev_positives.json --out $NR1_OUT/baseline/$dt --device cuda --score-dtype $dt
done
$NR1_PY -B site/step0_fullDEV_diag.py
$NR1_PY -I -B site/write_binding.py
echo "$(T) EXTRACT_JOB_DONE"
