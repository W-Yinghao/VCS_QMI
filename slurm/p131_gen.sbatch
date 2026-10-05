#!/usr/bin/env bash
#SBATCH --job-name=p131_gen
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P131 step 1 (CPU): regenerate the CIFAR-10-C-style corruptions from the official test images (owner authorisation 2026-10-04,
# reporting only) -> $P131_REGEN_DIR (default outputs/P131_c10c_regen).  Env: N_IMAGES (smoke), WORKERS.  Normal QOS only.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P131 generate: regen=${P131_REGEN_DIR:-$OUTPUT_ROOT/P131_c10c_regen} n=${N_IMAGES:-all}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VCS_P131_C10C=1 "$PYTHON" scripts/p131_c10c.py generate --owner-authorised-p131 --workers "${WORKERS:-15}" --n-images "${N_IMAGES:-0}"; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
