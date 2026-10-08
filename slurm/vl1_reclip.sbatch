#!/usr/bin/env bash
#SBATCH --job-name=vl1_reclip
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# ReCLIP reproduction (official code, allenai/reclip @ 3ed4f47 + documented import patch) with ground-truth boxes on rebuilt RefCOCOg-UMD inputs.
#   sbatch slurm/vl1_reclip.sbatch <val|dev|test> <parse|baseline>      (val = reproduction check against the paper; dev = our development role)
E=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; X=/projects/EEG-foundation-model/yinghao/external/reclip_patched
D=/projects/EEG-foundation-model/yinghao/datasets/refcocog_umd; O=/home/infres/yinwang/CS_QMI/outputs/VL1_reclip; mkdir -p $O
SPLIT=$1; METHOD=$2; export CLIP_CACHE=/projects/EEG-foundation-model/yinghao/models/openai_clip
cd $X && $E/bin/python main.py --input_file $D/reclip_inputs/refcocog_umd_${SPLIT}.jsonl --image_root $D/train2014_links --method $METHOD \
  --gradcam_alpha 0.5 0.5 --box_method_aggregator sum --clip_model RN50x16,ViT-B/32 --box_representation_method crop,blur --output_file $O/reclip_${METHOD}_${SPLIT}.json \
  > $O/reclip_${METHOD}_${SPLIT}.log 2>&1; rc=$?
tail -3 $O/reclip_${METHOD}_${SPLIT}.log; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
