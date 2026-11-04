#!/usr/bin/env bash
# usage: sbatch cc_extract.sbatch <COND>   (CC152K_PARTIAL_*; data under /projects/.../cc152k_raw/nr1/<COND>)
#SBATCH --job-name=nr1_cc_extract
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr1_20261104/sbatch/common_env.sh
C=$1; D=/projects/EEG-foundation-model/yinghao/data/cc152k_raw/nr1/$C
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
for split in dev fit; do
  if [ ! -f $D/cache/$split/manifest.json ]; then s=$SECONDS
    $NR1_PY -B -m nr1.extract_siglip2 --images $D/data/${split}_images.jsonl --texts $D/data/${split}_texts.jsonl \
      --split $split --out $D/cache/$split --revision $NR1_REV --device cuda --batch-size 32 --local-only
    echo "{\"split\":\"$split\",\"seconds\":$((SECONDS-s)),\"job\":\"$SLURM_JOB_ID\",\"node\":\"$(hostname)\"}" > $D/cache/$split/EXTRACT_COST.json
  fi
done
du -sb $D/cache/*
[ -f $D/calibration.json ] || NR1_FEATURE_RAM=global $NR1_PY -B -m nr1.calibrate --cache $D/cache/fit --observed $D/data/observed_$C.tsv --out $D/calibration.json
cat $D/calibration.json
[ -f $D/baseline/METRICS.json ] || $NR1_PY -B -m nr1.evaluate --cache $D/cache/dev --truth $D/evaluator_only/dev_positives.json --out $D/baseline --device cuda --score-dtype float64
$NR1_PY -B site/step0_diag_generic.py --dev-cache $D/cache/dev --calibration $D/calibration.json --out $D/STEP0_DIAG.json
echo "CC_EXTRACT_DONE $C"
