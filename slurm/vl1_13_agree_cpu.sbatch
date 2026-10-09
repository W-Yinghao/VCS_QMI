#!/usr/bin/env bash
#SBATCH --job-name=vl1_13_agree
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL1-13 gate: DEV argmax agreement of the ReCLIP-feature cache with ReCLIP's own IPS-only output (CPU; vl_baselines venv).
E=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; cd /home/infres/yinwang/CS_QMI/ssl_pilot; rc=0
CUDA_VISIBLE_DEVICES= $E/bin/python scripts/vl1_13_reclip_features.py agree || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
