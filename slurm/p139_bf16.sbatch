#!/usr/bin/env bash
#SBATCH --job-name=p139_bf16
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p139_bf16_%j.out
# P139 bf16 vs fp32 from a 200-step fp32-trained snapshot (R50, b 64 x 2 views), baselines fp32 repeat and fp32 + input noise; no checkpoints.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
set -o pipefail
banner "P139 bf16 precision check"
"$PYTHON" scripts/p139_imagenet_pipeline.py bf16 --out reports/P139/bf16_${SLURM_JOB_PARTITION:-x}_${SLURM_JOB_ID}.json 2>&1 | grep -vE "Warning|setter|warn" | tail -6; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
