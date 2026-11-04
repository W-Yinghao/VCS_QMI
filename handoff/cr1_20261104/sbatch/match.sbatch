#!/usr/bin/env bash
# usage: sbatch --job-name=cr1_match_<COND>_<SRC> match.sbatch <COND> <SRC> <MODE...>   (CPU, no gradients)
#SBATCH --partition=CPU,cpu-high
#SBATCH --qos=normal
#SBATCH --nice=50
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cr1_20261104/logs/%x_%j.out
set -euo pipefail
export OMP_NUM_THREADS=4
source /home/infres/yinwang/CS_QMI/handoff/cr1_20261104/sbatch/common_env.sh
C=$1; S=$2; shift 2
echo "JOB $SLURM_JOB_ID $(hostname) $(date -u +%FT%TZ)"
for M in "$@"; do
  $CR1_PY -I -B site/cr1_prod.py match --bank $CR1_OUT/banks/$C/fit_bank.npz --scores $CR1_OUT/$C/scores/$S.npz \
     --mode $M --out $CR1_OUT/$C/relations/${S}_${M}.npz --device cpu
  echo "DONE ${S}_${M} $(date -u +%FT%TZ)"
done
echo "EXIT_OK $(date -u +%FT%TZ)"
