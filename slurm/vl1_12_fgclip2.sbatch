#!/usr/bin/env bash
#SBATCH --job-name=vl1_12_fgclip2
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL1-12: FG-CLIP 2 Base official region path — COCO val2017 GT-box check (paper 74.9), then the RefCOCOg train-side feature cache.
E=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; export HF_HUB_OFFLINE=1; cd /home/infres/yinwang/CS_QMI/ssl_pilot; rc=0
$E/bin/python scripts/vl1_12_fgclip2_features.py check --n-images 2000 || rc=1
$E/bin/python scripts/vl1_12_fgclip2_features.py cache || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
