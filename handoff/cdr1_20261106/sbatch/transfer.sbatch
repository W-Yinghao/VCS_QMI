#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/cdr1_20261106/logs/%x_%j.out
set -euo pipefail
SEED=$1
PY=/home/infres/yinwang/CS_QMI/env_nr1/bin/python
T=/home/infres/yinwang/CS_QMI/handoff/cdr1_20261106/site/run_transfer.py
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "host $(hostname) start $(date -u +%FT%TZ) seed=$SEED test_out=${CDR1_TEST_OUT:-none}"
for S in cc_build predict evaluate done; do $PY -I -B $T --seed $SEED --stage $S; done
echo "TRANSFER_JOB_DONE $SEED $(date -u +%FT%TZ)"
