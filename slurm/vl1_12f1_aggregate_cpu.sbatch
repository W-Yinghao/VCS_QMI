#!/usr/bin/env bash
#SBATCH --job-name=vl1_12f1_aggregate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL1-12 addendum 3 (FG-CLIP v1) frozen readings.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; rc=0
echo "== table"; "$PYTHON" scripts/vl1_10_aggregate.py --dir outputs/VL1_12_fgclip1 --out reports/VL1/VL1_12f1_table.json --ns all --routes vcs js softmax rff || rc=1
echo "== FG-CLIP v1 − CLIP"; "$PYTHON" scripts/vl1_12_compare.py --a clip:VL1_10 --b fgclip1:VL1_12_fgclip1 --out reports/VL1/VL1_12f1_fgclip1_minus_clip.json || rc=1
echo "== strata"; "$PYTHON" scripts/vl1_10_strata.py --dir outputs/VL1_12_fgclip1 --features /projects/EEG-foundation-model/yinghao/datasets/refcocog_umd/features_fgclip1_base/train_side_features.pt --routes vcs js softmax --out reports/VL1/VL1_12f1_strata.json || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
