#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --nice=30
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r24_neon_host_20261027/logs/%x_%j.out
# usage: sbatch --job-name=r24_neon_<tag> --time=H:MM:SS gpu.sbatch <dhr_run.py args>
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd /home/infres/yinwang/CS_QMI/handoff/r24_neon_host_20261027
/projects/EEG-foundation-model/yinghao/envs/dhr/bin/python -I -u dhr_run.py "$@"
