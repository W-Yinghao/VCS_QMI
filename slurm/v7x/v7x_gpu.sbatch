#!/usr/bin/env bash
#SBATCH --job-name=v7x_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V7X GPU jobs (reports/V7/V7X_*): env envs/v7_fga (torch 2.9 cu128 -> RTX6000PRO / H100 / L40S).  Positional arguments only.
#   x20-smoke                 X20 extractor on 64 images of each role (code check + invariant + timing; no result is used)
#   x20                       X20 common features: FIT / CAL_SELECT / CAL_PROB / DEV
#   x20-rank                  X20 common features of the GenAI-Rank candidates (parser v3 elements)
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
E=/projects/EEG-foundation-model/yinghao/envs/v7_fga; PY="$E/bin/python -I"; BIG=/projects/EEG-foundation-model/yinghao/outputs_v7
export TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) branch=$(git branch --show-current) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts/v7x tests/v7x slurm/v7x | wc -l)"
echo "code: $(cat scripts/v7x/*.py tests/v7x/*.py slurm/v7x/*.sbatch 2>/dev/null | sha256sum | cut -c1-16) (V7X scripts, tests, wrappers); v1 code: $(cat scripts/v7/*.py tests/v7/*.py slurm/v7/*.sbatch | sha256sum | cut -c1-16)"
what="$1"; shift
case "$what" in
  x20-smoke) $PY scripts/v7x/x20_features.py extract --out $BIG/x20_smoke --roles FIT CAL_SELECT CAL_PROB DEV --limit 64; rc=$?;;
  x20) $PY scripts/v7x/x20_features.py extract --out $BIG/x20; rc=$?;;
  x20-rank) $PY scripts/v7x/x20_features.py extract-rank --out $BIG/x20; rc=$?;;
  *) echo "unknown mode $what"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
