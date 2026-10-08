#!/usr/bin/env bash
#SBATCH --job-name=p139_check_cpu
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p139_check_cpu_%j.out
# P139 device-independent checks on CPU: (a) fixed-parameter GPU-stage code vs torchvision tensor ops vs P132 PIL ops on 600 real crops;
# (b) random-parameter output-summary distributions, P132 PIL pipeline vs CPU-crop + augmentation stage (2000 images x 2 views).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
set -o pipefail
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
banner "P139 check (CPU part)"
"$PYTHON" scripts/p139_imagenet_pipeline.py check --device cpu --out reports/P139/check_cpu_${SLURM_JOB_ID}.json 2>&1 | grep -v Warning | tail -8; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
