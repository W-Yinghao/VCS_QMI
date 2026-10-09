#!/usr/bin/env bash
#SBATCH --job-name=vl1_strata
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL1-10 / VL1-12 secondary reading 5 (strata) + fit-time column, evaluation only from saved checkpoints; main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; rc=0
echo "== CLIP (VL1-10)"; "$PYTHON" scripts/vl1_10_strata.py --ns 1000 4000 all || rc=1
echo "== FG-CLIP 2 (VL1-12)"; "$PYTHON" scripts/vl1_10_strata.py --dir outputs/VL1_12 --features /projects/EEG-foundation-model/yinghao/datasets/refcocog_umd/features_fgclip2_base/train_side_features.pt --routes vcs js softmax --out reports/VL1/VL1_12_strata.json || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
