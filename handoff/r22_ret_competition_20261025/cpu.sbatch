#!/usr/bin/env bash
#SBATCH --job-name=r22_retcomp_fit
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Fits (stage c0 read guard: no labels) for all seeds, then the label-reading analysis as a separate final step.
set -u
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
H=/home/infres/yinwang/CS_QMI/handoff/r22_ret_competition_20261025
T=/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2p_c/v9_n2p/c_server/trace_open.py
rc=0
for s in 901 902 903; do
  $PY -I -B $T --stage c0 --out $H/traces/fit_s${s}_${SLURM_JOB_ID}.json -- $H/compete.py fit --seed $s || rc=$?
done
[ $rc -eq 0 ] && $PY -I -B $H/compete.py analyze || rc=$?
echo "end=$(date -u +%FT%TZ) rc=$rc"; exit $rc
