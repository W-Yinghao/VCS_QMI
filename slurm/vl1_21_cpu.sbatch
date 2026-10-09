#!/usr/bin/env bash
#SBATCH --job-name=vl1_21_excl
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
cd /home/infres/yinwang/CS_QMI/ssl_pilot; /projects/EEG-foundation-model/yinghao/envs/vl_baselines/bin/python scripts/vl1_21_detector.py exclusions; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
