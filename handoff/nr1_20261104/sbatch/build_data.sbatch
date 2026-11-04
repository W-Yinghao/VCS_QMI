#!/usr/bin/env bash
#SBATCH --job-name=nr1_build_data
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
/home/infres/yinwang/CS_QMI/env_nr1/bin/python -I -B /home/infres/yinwang/CS_QMI/handoff/nr1_20261104/code/site/build_flickr_data.py --out /home/infres/yinwang/CS_QMI/outputs_nr1/NR1_20261104
