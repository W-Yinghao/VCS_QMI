#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=r33_feeder
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r33_baselines_20261102/logs/%x_%j.out
exec bash /home/infres/yinwang/CS_QMI/handoff/r33_baselines_20261102/sbatch/r33_feed.sh
