#!/usr/bin/env bash
#SBATCH --qos=normal
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Package step 2: python tests/check_cpu.py (real PyTorch CPU checks on the shipped provenance snapshot).  The script rewrites
# results/CPU_VERIFICATION.json inside the package; the report is copied to the round output and the shipped file restored.
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset PYTHONPATH
WT=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_appround
PKG=$WT/v10/app_round_20261021/package
OUT=/home/infres/yinwang/CS_QMI/handoff/v10_app_round_20261021/cpu_check
mkdir -p "$OUT"
echo "host=$(hostname) start=$(date -u +%FT%TZ) job=${SLURM_JOB_ID} commit=$(git -C "$WT" rev-parse HEAD)"
cd "$PKG"
set +e
/home/infres/yinwang/CS_QMI/env/bin/python -B -s tests/check_cpu.py; rc=$?
set -e
cp results/CPU_VERIFICATION.json "$OUT/CPU_VERIFICATION_${SLURM_JOB_ID}.json"
git -C "$WT" checkout -- v10/app_round_20261021/package/results/CPU_VERIFICATION.json
echo "end=$(date -u +%FT%TZ) exit=${rc}"
exit "$rc"
