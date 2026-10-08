#!/usr/bin/env bash
#SBATCH --job-name=figures_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; PYTHON=/projects/EEG-foundation-model/yinghao/envs/vl_baselines/bin/python; rc=0  # matplotlib lives in the baseline venv; the main env stays unchanged
"$PYTHON" scripts/fig_decisive_estimator.py || rc=1
"$PYTHON" scripts/ablation_table.py || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
