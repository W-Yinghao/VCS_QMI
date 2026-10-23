#!/usr/bin/env bash
#SBATCH --partition=cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R22-B CPU checks of the posterior code (tests/test_r22_posterior.py) in the NCR env.
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v9_r22b
OUT=${1:-/home/infres/yinwang/CS_QMI/outputs_v9_r22/ncr_vcs/validation/CPU_CHECKS.json}
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git -C $R rev-parse --short HEAD) job=$SLURM_JOB_ID"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -I $R/v9_r22/ncr_vcs/tests/test_r22_posterior.py --out "$OUT"
rc=$?; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
