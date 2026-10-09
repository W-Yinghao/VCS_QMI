#!/usr/bin/env bash
#SBATCH --job-name=vl1_13_aggregate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL1-13 frozen readings (ReCLIP isolation features).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; rc=0
echo "== table"; "$PYTHON" scripts/vl1_10_aggregate.py --dir outputs/VL1_13 --out reports/VL1/VL1_13_table.json --ns all --routes vcs js softmax rff || rc=1
echo "== ReCLIP features − CLIP crops (context)"; "$PYTHON" scripts/vl1_12_compare.py --a clip:VL1_10 --b reclip:VL1_13 --out reports/VL1/VL1_13_reclip_minus_clip.json || rc=1
echo "== strata"; "$PYTHON" scripts/vl1_10_strata.py --dir outputs/VL1_13 --features /projects/EEG-foundation-model/yinghao/datasets/refcocog_umd/features_reclip_ips/train_side_features.pt --routes vcs js softmax --out reports/VL1/VL1_13_strata.json || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
