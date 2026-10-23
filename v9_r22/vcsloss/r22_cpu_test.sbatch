#!/usr/bin/env bash
#SBATCH --job-name=r22_vl_cputest
#SBATCH --partition=cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R22-A / A2 / A3 CPU checks (test_r22.py, test_r22_a2.py, test_r22_a3.py; synthetic fixtures, plus the locked N2F weight files read by
# test_r22_a3.py).  R22_DIR overrides the code directory.
set -u
D=${R22_DIR:-/home/infres/yinwang/CS_QMI/ssl_pilot_v9_r22a/v9_r22/vcsloss}
OUT=/home/infres/yinwang/CS_QMI/outputs_v9_r22/vcsloss/unit_test; mkdir -p "$OUT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "host=$(hostname) job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ) head=$(git -C "$D" rev-parse HEAD)"
cd "$D" && /home/infres/yinwang/CS_QMI/env/bin/python -I -B -m pytest -q -p no:cacheprovider test_r22.py test_r22_a2.py test_r22_a3.py \
  --junitxml="$OUT/junit_${SLURM_JOB_ID:-nojob}.xml"; rc=$?
echo "end=$(date -u +%FT%TZ) rc=$rc"; exit $rc
