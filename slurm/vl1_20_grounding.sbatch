#!/usr/bin/env bash
#SBATCH --job-name=vl1_20
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL1-20 Table C full-grounding baselines; arguments = one or more "<cmd> <args>" groups separated by --then (vl_baselines venv).
E=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home
cd /home/infres/yinwang/CS_QMI/ssl_pilot; rc=0
while [ $# -gt 0 ]; do args=(); while [ $# -gt 0 ] && [ "$1" != "--then" ]; do args+=("$1"); shift; done; [ "$1" = "--then" ] && shift
  $E/bin/python scripts/vl1_20_grounding.py "${args[@]}" || rc=1; done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
