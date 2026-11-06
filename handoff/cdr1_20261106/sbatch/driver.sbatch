#!/bin/bash
#SBATCH --job-name=cdr1_driver
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cdr1_20261106/logs/%x_%j.out
/home/infres/yinwang/CS_QMI/env_nr1/bin/python -I -B /home/infres/yinwang/CS_QMI/handoff/cdr1_20261106/site/driver.py "$@"
