#!/bin/bash
#SBATCH --job-name=r26_cpu
#SBATCH --qos=normal
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r26_dist_20261027/logs/%x_%j.out
cd /home/infres/yinwang/CS_QMI/handoff/r26_dist_20261027/code
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -u r26_cpu_check.py /home/infres/yinwang/CS_QMI/handoff/r26_dist_20261027/CPU_CHECK.json
