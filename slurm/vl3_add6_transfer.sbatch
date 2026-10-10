#!/usr/bin/env bash
#SBATCH --job-name=vl3_add6_transfer
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL3 add. 6: cross-dataset transfer of the VL3 checkpoints (scripts/vl3_add6_transfer.py); args passed through.
E=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; cd /home/infres/yinwang/CS_QMI/ssl_pilot
echo "dataset=${VL_DATASET:-refcocog} host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date -u +%FT%TZ)"
$E/bin/python scripts/vl3_add6_transfer.py "$@" --workers 10; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
