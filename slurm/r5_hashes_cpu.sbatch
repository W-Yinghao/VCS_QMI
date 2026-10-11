#!/usr/bin/env bash
#SBATCH --job-name=r5_hashes
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r5 registration hashes (scripts/r5_hashes.py) for both datasets, sequentially (one JSON); VL env, offline.
E=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; cd /home/infres/yinwang/CS_QMI/ssl_pilot; rc=0
for d in refcocog refcoco+; do VL_DATASET=$d $E/bin/python scripts/r5_hashes.py || rc=1; done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
