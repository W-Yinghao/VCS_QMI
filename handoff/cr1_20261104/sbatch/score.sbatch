#!/usr/bin/env bash
# usage: sbatch --job-name=cr1_score_<COND>_<SRC> score.sbatch <COND> <SRC>
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr1_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr1_20261104/sbatch/common_env.sh
C=$1; S=$2
echo "JOB $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)"
$CR1_PY -I -B site/cr1_prod.py score --bank $CR1_OUT/banks/$C/fit_bank.npz --prepared $CR1_OUT/$C/prepared \
   --source $S --models-root $CR1_OUT/$C/experts --out $CR1_OUT/$C/scores/$S.npz --device cuda
echo "EXIT_OK $(date -u +%FT%TZ)"
