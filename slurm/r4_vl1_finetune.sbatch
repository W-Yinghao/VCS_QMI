#!/usr/bin/env bash
#SBATCH --job-name=r4_vl1
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=23:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R4-VL1 proposal-aware full fine-tuning (scripts/r4_vl1_finetune.py); VL_DATASET exported; args passed through.
E=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; cd /home/infres/yinwang/CS_QMI/ssl_pilot
echo "dataset=${VL_DATASET} host=$(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ)"
$E/bin/python scripts/r4_vl1_finetune.py "$@" --workers 10; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
