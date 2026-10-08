#!/usr/bin/env bash
#SBATCH --job-name=vl1_00_clip
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL1-00 raw-CLIP baseline on the RefCOCOg (UMD) training side (FIT / CAL / DEV); official val / test untouched.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export HF_HUB_OFFLINE=1
"$PYTHON" scripts/vl1_00_clip_baseline.py --roles FIT,CAL,DEV --workers 14; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
