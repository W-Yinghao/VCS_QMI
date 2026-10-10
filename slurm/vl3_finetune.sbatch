#!/usr/bin/env bash
#SBATCH --job-name=vl3_ft
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL3 full fine-tuning (scripts/vl3_finetune.py); VL_DATASET exported; args = one or more fitter argument groups separated by --then.
E=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; cd /home/infres/yinwang/CS_QMI/ssl_pilot; rc=0
echo "dataset=${VL_DATASET:-refcocog} host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date -u +%FT%TZ)"
while [ $# -gt 0 ]; do args=(); while [ $# -gt 0 ] && [ "$1" != "--then" ]; do args+=("$1"); shift; done; [ "$1" = "--then" ] && shift
  $E/bin/python scripts/vl3_finetune.py "${args[@]}" --workers 10 || rc=1; done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
