#!/usr/bin/env bash
# Shared environment for all VCS-QMI SSL pilot jobs.  Source from sbatch scripts.
# Paths were resolved in P0 (see reports/P0_preflight.md); the trainer refuses unresolved placeholders.
export REPO_ROOT=/home/infres/yinwang/CS_QMI/ssl_pilot
export DATA_ROOT=/home/infres/yinwang/CS_QMI/data/cifar10
export OUTPUT_ROOT=/home/infres/yinwang/CS_QMI/outputs
export MANIFEST_ROOT=/home/infres/yinwang/CS_QMI/manifests
export PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$OUTPUT_ROOT" "$MANIFEST_ROOT" "$REPO_ROOT/reports"
cd "$REPO_ROOT" || exit 2
banner() {
  echo "=================================================================="
  echo " $1"
  echo " host=$(hostname) job=${SLURM_JOB_ID:-NA} partition=${SLURM_JOB_PARTITION:-NA} gpus=${CUDA_VISIBLE_DEVICES:-NA} start=$(date -u +%FT%TZ)"
  echo " commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null) dirty=$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | wc -l)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
  echo "=================================================================="
}
# Forward SIGTERM (sent by SLURM at walltime, --signal=B:TERM@120) to the python child so it can record STOPPED_BUDGET.
run_fg() {
  "$@" &
  local pid=$!
  trap 'echo "[slurm] TERM received, forwarding to $pid"; kill -TERM $pid 2>/dev/null' TERM INT
  wait $pid
  local rc=$?
  if [ $rc -gt 128 ]; then wait $pid; rc=$?; fi   # first wait returned because of the trap; collect the child's real exit code
  trap - TERM INT
  return $rc
}
