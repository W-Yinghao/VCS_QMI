#!/usr/bin/env bash
#SBATCH --job-name=v9n2m_drvprobe
#SBATCH --cpus-per-task=1
#SBATCH --mem=500M
#SBATCH --time=00:03:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Read-only: print the NVIDIA kernel-module (driver) version of the node this CPU-only job lands on.
# The N2M runner refuses any allocation whose driver differs from the frozen target environment, so the
# seed jobs must be restricted to RTX6000PRO nodes with the same driver as the CUDA verification job.
echo "host=$(hostname) utc=$(date -u +%FT%TZ)"; cat /proc/driver/nvidia/version 2>&1 | head -2; uname -r
