#!/usr/bin/env bash
#SBATCH --job-name=vl2_feat
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL2 feature caches: VL_DATASET (exported) selects RefCOCO / RefCOCO+; ENVPY = main env or the vl_baselines venv; args = extractor + its args.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; cd /home/infres/yinwang/CS_QMI/ssl_pilot; rc=0
echo "dataset=$VL_DATASET host=$(hostname) start=$(date -u +%FT%TZ)"
"$ENVPY" "$@" || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
