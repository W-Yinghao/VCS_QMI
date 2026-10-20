#!/usr/bin/env bash
# CPU-only RuLSIF_exact_v2 work (FIT design evidence, FIT functional gate, CPU tests).
# Runs from a snapshot of one committed revision, never from the live worktree.
#SBATCH --job-name=v10r2_cpu
#SBATCH --partition=cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export V10_TEST_DEVICE=cpu
: "${SLURM_JOB_ID:?Slurm required}"
: "${V10R2_REPO:?git worktree holding the commit}"
: "${V10R2_COMMIT:?pinned commit required}"
SNAP="${TMPDIR:-/tmp}/v10r2_${SLURM_JOB_ID}"
mkdir -p "$SNAP"
trap 'rm -rf "$SNAP"' EXIT
git -C "$V10R2_REPO" archive "$V10R2_COMMIT" src v10 scripts tests slurm | tar -x -C "$SNAP"
echo "v10r2 snapshot commit=$(git -C "$V10R2_REPO" rev-parse "$V10R2_COMMIT") host=$(hostname) start=$(date -u +%FT%TZ)"
export PYTHONPATH="$SNAP/src:$SNAP"
cd "$SNAP"
set +e
/home/infres/yinwang/CS_QMI/env/bin/python -u "$@"
rc=$?
set -e
echo "v10r2 exit=$rc end=$(date -u +%FT%TZ)"
exit "$rc"
