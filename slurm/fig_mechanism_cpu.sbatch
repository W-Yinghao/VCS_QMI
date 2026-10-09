#!/usr/bin/env bash
#SBATCH --job-name=fig_mechanism
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; PYTHON=/projects/EEG-foundation-model/yinghao/envs/vl_baselines/bin/python; rc=0  # matplotlib lives in the baseline venv
"$PYTHON" scripts/fig_mechanism.py || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
