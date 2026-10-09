#!/usr/bin/env bash
#SBATCH --job-name=vl1_13_feat
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL1-13 (DRAFT): ReCLIP isolation features with ReCLIP's own preprocessing — smoke (20 images), full train-side cache, DEV agreement gate.
E=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; export HF_HUB_OFFLINE=1; cd /home/infres/yinwang/CS_QMI/ssl_pilot
$E/bin/python scripts/vl1_13_reclip_features.py cache --limit 20 --workers 8 || { echo "exit=1 (smoke) end=$(date -u +%FT%TZ)"; exit 1; }
rc=0; $E/bin/python scripts/vl1_13_reclip_features.py cache --workers 15 || rc=1
[ $rc = 0 ] && { $E/bin/python scripts/vl1_13_reclip_features.py agree || rc=1; }
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
