#!/usr/bin/env bash
#SBATCH --job-name=rev9f_direct_h100_eval
#SBATCH --partition=H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# REV9 execution note 3: the frozen D1-30 evaluator on the H100 direct_lr0.0001_s0 I1 / I2 units (8 configs per benchmark), same runner as eval-full.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf
E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm; rc=0
for b in gqa_dev textvqa_val chartqa_val; do
  $E/bin/python scripts/r7_d1_30_run.py --sets-dir /home/infres/yinwang/CS_QMI/outputs/R7_D1/d1_30/sets --bench $b --configs-file /home/infres/yinwang/CS_QMI/outputs/REV9_FULL_H100/configs/configs_$b.txt --shard 0/1 --out /home/infres/yinwang/CS_QMI/outputs/REV9_FULL_H100/d1_30/pred     --selectors-dir /home/infres/yinwang/CS_QMI/outputs/REV9_FULL_H100/selectors --pca /home/infres/yinwang/CS_QMI/outputs/R7_D1_FULL/cache/pca.pt || rc=1
done; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
