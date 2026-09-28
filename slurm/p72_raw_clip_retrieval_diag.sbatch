#!/usr/bin/env bash
#SBATCH --job-name=p72_raw_clip_retrieval_diag
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p72_raw_clip_retrieval_diag_%j.out
cd /home/infres/yinwang/CS_QMI/ssl_pilot && /home/infres/yinwang/CS_QMI/env/bin/python scripts/r2_raw_clip_retrieval.py
