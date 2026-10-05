#!/usr/bin/env bash
#SBATCH --job-name=p132_gpu_tp
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p132_gpu_tp_%j.out
# P132 ImageNet preflight, single-GPU throughput (ResNet-50 224 px, bf16 autocast): A-P3 all-view tokens and multi-view SimCLR at 2 views
# (128 images / GPU) and 4 views (64 images / GPU) on synthetic input (GPU ceiling), plus A-P3 2 / 4 views on the real multi-view loader
# (14 workers); 50 steps per cell, 300 steps in total; no checkpoints.  Submit once per GPU type:
#   sbatch --partition=RTX6000PRO slurm/p132_gpu_throughput.sbatch   (then H100, L40S)
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
set -o pipefail
R=$REPO_ROOT/reports/P132; mkdir -p "$R"
banner "P132 GPU throughput on ${SLURM_JOB_PARTITION:-?} -> $R"
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader
"$PYTHON" scripts/p132_imagenet_preflight.py gpu --out "$R/gpu_${SLURM_JOB_PARTITION:-x}_${SLURM_JOB_ID}.json" --workers 14 2>&1 | tail -12; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
