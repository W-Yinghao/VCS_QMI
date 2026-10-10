#!/usr/bin/env bash
#SBATCH --job-name=vl1_21_eval
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL1-21 clean detector: gate D (COCO val2017 bbox AP >= 33) and the CAL / DEV / val proposals of the three datasets (scripts/vl1_21_detector.py eval).
E=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; export TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home OMP_NUM_THREADS=4
cd /home/infres/yinwang/CS_QMI/ssl_pilot; echo "host $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start $(date -u +%FT%TZ)"
$E/bin/python scripts/vl1_21_detector.py eval; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
