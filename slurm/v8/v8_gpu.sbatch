#!/usr/bin/env bash
#SBATCH --job-name=v8_gpu
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V8 GPU jobs (reports/V8_UTILITY).  env envs/v7_fga.  Positional arguments only.
#   smoke         every arm: 20 steps x 5 epochs + CAL_SELECT eval on 64 images (outputs_v8/smoke/<arm>), then one arm's export on CAL_SELECT
#   epoch-timing  U-CS, one full FIT epoch + the full CAL_SELECT eval (outputs_v8/timing)
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v8; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY="/projects/EEG-foundation-model/yinghao/envs/v7_fga/bin/python -I"; O=/projects/EEG-foundation-model/yinghao/outputs_v8
export TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) branch=$(git branch --show-current) commit=$(git rev-parse --short HEAD)"
what="$1"; shift; rc=0
case "$what" in
  smoke) for arm in U-CS U-BCE U-RPS U-CE U-MSE U-L1 U-RANK U-LIST; do
           $PY scripts/v8/v8_train.py train --arm $arm --lr 1e-5 --seed 0 --unit-dir $O/smoke/$arm --max-steps 20 --eval-n 64 --workers 6 || rc=1; done
         $PY scripts/v8/v8_train.py export --unit-dir $O/smoke/U-CS --roles CAL_SELECT || rc=1;;
  epoch-timing) $PY scripts/v8/v8_train.py train --arm U-CS --lr 1e-5 --seed 0 --unit-dir $O/timing/U-CS --epochs 1 --workers 8; rc=$?;;
  *) echo "usage: smoke | epoch-timing"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
