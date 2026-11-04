#!/usr/bin/env bash
# usage: sbatch --job-name=cr1_est_<COND>_<OBJ> est.sbatch <COND> <OBJ>   (six leave-two-fold-out experts, sequential)
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --nice=50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr1_20261104/logs/%x_%j.out
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/cr1_20261104/sbatch/common_env.sh
C=$1; O=$2; B=$CR1_OUT/banks/$C/fit_bank.npz; PR=$CR1_OUT/$C/prepared
echo "JOB $SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader|head -1) $(date -u +%FT%TZ)"
for E in 01 02 03 12 13 23; do
  U=CR1_${C}_EST_${O}_E${E}_s1001; D=$CR1_OUT/$C/experts/$O/E$E; mkdir -p $D
  echo "$SLURM_JOB_ID $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) $(date -u +%FT%TZ)" >> $D/ATTEMPTS.txt
  [ -f $D/DONE.json ] || $CR1_PY -I -B site/cr1_prod.py fit-estimator --bank $B --prepared $PR --unit $U --out $D --device cuda
  echo "DONE $U $(date -u +%FT%TZ)"
done
echo "EXIT_OK $(date -u +%FT%TZ)"
