#!/usr/bin/env bash
#SBATCH --job-name=vl1_21_det
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=36
#SBATCH --mem=160G
#SBATCH --time=23:59:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL1-21 clean detector: train (torchrun, NGPU GPUs; resumes from last.pt) — args passed to the train subcommand (e.g. --smoke).
E=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; export TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home OMP_NUM_THREADS=4
cd /home/infres/yinwang/CS_QMI/ssl_pilot; NG=${NGPU:-4}; echo "host $(hostname) gpus $NG start $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name --format=csv,noheader | head -1
[ -f /home/infres/yinwang/CS_QMI/outputs/VL1_21/exclusions.json ] || $E/bin/python scripts/vl1_21_detector.py exclusions || { echo "exit=1 (exclusions) end=$(date -u +%FT%TZ)"; exit 1; }
$E/bin/torchrun --standalone --nproc_per_node $NG scripts/vl1_21_detector.py train --workers 8 "$@"; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
