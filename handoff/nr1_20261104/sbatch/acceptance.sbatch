#!/usr/bin/env bash
#SBATCH --job-name=nr1_accept
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/nr1_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/nr1_20261104/sbatch/common_env.sh
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $NR1_PY -B -m pytest -q -p no:warnings -p no:cacheprovider tests 2>&1 | tail -3
$NR1_PY -B site/acceptance.py --out $NR1_OUT/acceptance_${SLURM_JOB_ID} --data $NR1_OUT/data --revision $NR1_REV --snapshot $NR1_SNAPSHOT
