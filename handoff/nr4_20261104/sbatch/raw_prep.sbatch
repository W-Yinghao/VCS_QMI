#!/usr/bin/env bash
# usage: sbatch --job-name=nr4_raw_<ds>_<split> raw_prep.sbatch <dataset> <split>
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr4_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr4_20261104/sbatch/common_env.sh
export OMP_NUM_THREADS=1
$NR1_PY -B -m nr4.raw_prep --dataset $1 --split $2 --procs 16
