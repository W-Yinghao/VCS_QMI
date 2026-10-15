#!/usr/bin/env bash
#SBATCH --job-name=v9_n2_precheck
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --nice=400
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-N2 pre-registered precheck gate on the real Flickr30K precomp data (owner 2026-10-16): one grouped GPU job.  For each seed in SEEDS: the common SGR
# teacher (5 warm-up-loss epochs on A-teacher-warmup; seconds / epoch and peak memory recorded = N2's first real cost number), teacher scores for every
# role and U, every weight method, the precheck tables for rho 0.4 and then rho 0 (the teacher is identical across rho — A is clean — and is copied,
# recorded in teacher_shared_from.json).  No student is trained, no dev / test read.  Env: fork A's v9_sgr (nltk punkt for the official tokeniser).
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v9; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=${V9_SGR_PY:-/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python}; OUT_ROOT=${OUT_ROOT:-/projects/EEG-foundation-model/yinghao/outputs_v9/n2}; mkdir -p "$OUT_ROOT"
DATA_ROOT=${DATA_ROOT:-/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data}; VOCAB=${VOCAB:-/projects/EEG-foundation-model/yinghao/code_v9/SGRAF_py3/vocab}
SEEDS=${SEEDS:-901 902 903}; RHOS=${RHOS:-0.4 0}   # tuning seed: SEEDS=917 (shared stage + merge groups for the tuning units; not a gate)
SNAP=$(mktemp -d /tmp/v9_n2_${SLURM_JOB_ID}_XXXX); trap "rm -rf $SNAP" EXIT; mkdir -p $SNAP/src/v9 $SNAP/scripts/v9 $SNAP/v9
cp -r src/v9/n2 $SNAP/src/v9/; cp src/v9/*.py $SNAP/src/v9/; cp scripts/v9/n2_run.py $SNAP/scripts/v9/; cp -r v9/package_v9_20261016 $SNAP/v9/; export PYTHONPATH=$SNAP/src
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) code=$(cat $SNAP/src/v9/n2/*.py $SNAP/scripts/v9/n2_run.py | sha256sum | cut -c1-16) seeds='$SEEDS' rhos='$RHOS'"
rc=0
for seed in $SEEDS; do for rho in $RHOS; do
  echo "[precheck] $(date -u +%FT%TZ) f30k rho=$rho seed=$seed"
  $PY $SNAP/scripts/v9/n2_run.py precheck --dataset f30k --rho $rho --seed $seed --data-root $DATA_ROOT --vocab $VOCAB --out-root $OUT_ROOT --host-lock reports/V9/HOST_LOCK.yaml --device cuda; r=$?
  echo "[precheck-exit] rc=$r $(date -u +%FT%TZ)"; [ $r -ne 0 ] && rc=$r
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1
done; done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
