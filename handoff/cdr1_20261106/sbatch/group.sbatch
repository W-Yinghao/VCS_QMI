#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cdr1_20261106/logs/%x_%j.out
# usage: sbatch --job-name=cdr1_g_<ds>_<seed> --mem=.. group.sbatch <dataset> <seed>
set -euo pipefail
DS=$1; SEED=$2
PY=/home/infres/yinwang/CS_QMI/env_nr1/bin/python
G=/home/infres/yinwang/CS_QMI/handoff/cdr1_20261106/site/run_group.py
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "host $(hostname) start $(date -u +%FT%TZ) ds=$DS seed=$SEED test_out=${CDR1_TEST_OUT:-none}"
for S in curate fit predict evaluate d1 oracle_fit oracle_eval rho_fit rho_predict rho_eval integrity done; do
  $PY -I -B $G --dataset $DS --seed $SEED --stage $S
done
echo "GROUP_JOB_DONE $DS $SEED $(date -u +%FT%TZ)"
