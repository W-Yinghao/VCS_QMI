#!/usr/bin/env bash
#SBATCH --job-name=p139_check
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p139_check_%j.out
# P139 GPU checks: fixed-parameter equality (GPU vs torchvision tensor ops vs P132 PIL ops) on 600 real crops; random-parameter output-summary
# distributions (P132 PIL pipeline vs CPU-crop + GPU stage, 2000 images x 2 views); GPU-stage cost; bf16 vs fp32 loss / gradient (R50, 1 batch).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
set -o pipefail
banner "P139 check"
"$PYTHON" scripts/p139_imagenet_pipeline.py check --out reports/P139/check_${SLURM_JOB_PARTITION:-x}_${SLURM_JOB_ID}.json 2>&1 | tail -20; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
