#!/usr/bin/env bash
#SBATCH --job-name=v10d_cpu
#SBATCH --partition=cpu-high
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# CPU-only checks of the post-DEV diagnostics before any GPU submission (no GPU, no DEV read):
#   tests             pytest tests/v10_sensitivity (synthetic inputs; frozen-tree regressions in subprocesses)
#   d1 RUN_KEY        D1 dry run: real FIT/CAL inputs, one CAL core at step 736 on CPU, scratch output
#   d2                D2 dry run: one CAL core, frozen replay + variant on CPU, scratch output
#   d3 FAMILY         D3 dry run: manifest copy, frozen import, patch binding; the gate itself is not run
#   sequence [KEY..]  tests, d2, d3 for both families, then d1 for each KEY, in one allocation
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE
: "${SLURM_JOB_ID:?Slurm required}"
DIAG_ROOT=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_diag
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
export V10_SUPPLEMENT_JOB_START
V10_SUPPLEMENT_JOB_START=$(date +%s.%N)
step=${1:?tests|d1|d2|d3}; shift
echo "[v10d-cpu] step=$step args=$* job=$SLURM_JOB_ID host=$(hostname) utc=$(date -u +%FT%TZ)"
case "$step" in
  tests)
    cd "$DIAG_ROOT"
    PYTHONPATH="$DIAG_ROOT/src" "$PYTHON" -m pytest -q -p no:cacheprovider tests/v10_sensitivity "$@" ;;
  d1)
    source_root=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_native_recovery_20261019
    cd "$source_root"; PYTHONPATH="$source_root/src:$source_root" \
      "$PYTHON" -u "$DIAG_ROOT/scripts/v10_sensitivity/d1_comir_checkpoints.py" "$@" --dry-run ;;
  d2)
    source_root=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_pipeline_20261019
    cd "$source_root"; PYTHONPATH="$source_root/src:$source_root" \
      "$PYTHON" -u "$DIAG_ROOT/scripts/v10_sensitivity/d2_mind_wtd.py" --dry-run ;;
  d3)
    source_root=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_pipeline_20261019
    cd "$source_root"
    V10_VALIDATION_MANIFEST=/home/infres/yinwang/CS_QMI/handoff/v10_sensitivity_20261019/D3/validation/MANIFEST.json \
    V10_VALIDATION_SHA256=8befdabb0c2b6823e8c04c86763090fd2b383b28b5c02250c34fcae15eb1292b \
    PYTHONPATH="$source_root/src:$source_root" \
      "$PYTHON" -u "$DIAG_ROOT/scripts/v10_sensitivity/d3_gate.py" "$@" --dry-run ;;
  sequence)  # one allocation for all checks when the shared normal job cap is tight; runs every step, then fails if any failed
    rc=0
    bash "$0" tests || rc=1
    bash "$0" d2 || rc=1
    bash "$0" d3 MLP_VCS_perm8 || rc=1
    bash "$0" d3 'InfoNCE_host[J_perm1]' || rc=1
    for key in "$@"; do bash "$0" d1 "$key" || rc=1; done
    [ "$rc" -eq 0 ] || { echo "[v10d-cpu] sequence had a failing step" >&2; exit 1; } ;;
  *) echo "unknown step $step" >&2; exit 2 ;;
esac
echo "[v10d-cpu] step=$step rc=0 end=$(date -u +%FT%TZ)"
