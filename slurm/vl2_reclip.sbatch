#!/usr/bin/env bash
#SBATCH --job-name=vl2_reclip
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL2 ReCLIP (official code, same flags as slurm/vl1_reclip.sbatch) on RefCOCO / RefCOCO+ inputs built by scripts/vl2_reclip_inputs.py.
#   sbatch --export=ALL,DS=refcocoplus slurm/vl2_reclip.sbatch <val|dev_referred> <parse|baseline>
E=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; X=/projects/EEG-foundation-model/yinghao/external/reclip_patched
D=/projects/EEG-foundation-model/yinghao/datasets/${DS}_unc; O=/home/infres/yinwang/CS_QMI/outputs/VL2_reclip; mkdir -p $O; SPLIT=$1; METHOD=$2
export HF_HUB_OFFLINE=1 CLIP_CACHE=/projects/EEG-foundation-model/yinghao/models/openai_clip
cd $X && $E/bin/python main.py --input_file $D/reclip_inputs/${DS}_${SPLIT}.jsonl --image_root $D/train2014_links --method $METHOD \
  --gradcam_alpha 0.5 0.5 --box_method_aggregator sum --clip_model RN50x16,ViT-B/32 --box_representation_method crop,blur --output_file $O/reclip_${DS}_${METHOD}_${SPLIT}.json \
  > $O/reclip_${DS}_${METHOD}_${SPLIT}.log 2>&1; rc=$?
tail -3 $O/reclip_${DS}_${METHOD}_${SPLIT}.log; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
