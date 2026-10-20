#!/usr/bin/env bash
#SBATCH --qos=normal
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Hardware-amendment K8/K32 CUDA gate (execution / numerical gate, not scientific evidence).  One job per GPU type:
#   sbatch --partition=RTX6000PRO --job-name=v10k_hwgate_rtx --export=ALL,V10_CAPACITY_FREEZE=<base>,V10_CAPACITY_HW_FREEZE=<hw> hw_gate.sbatch
#   sbatch --partition=H100       --job-name=v10k_hwgate_h100 ...
# Writes handoff/v10_capacity_ext_20261019/hw_amendment_20261019/gpu_gates/<partition>_<job>.json; run_hw.py requires a PASS for
# RTX6000PRO / H100 before a unit may run there.  Same environment as run_hw.sbatch.
set -euo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE PYTHONPATH
: "${V10_CAPACITY_FREEZE:?base freeze sha required}" "${V10_CAPACITY_HW_FREEZE:?amendment freeze sha required}"
export V10_EXPECT_CODE=e08ee2c176475e2b V10_SUBMIT_COMMIT=b8f59ea825dc7a2064457d47a628bef09ca4e185
WT=/home/infres/yinwang/CS_QMI/ssl_pilot_v10_capacity_hw_20261019
GATE_OUT=/projects/EEG-foundation-model/yinghao/outputs_v10/capacity_ext_hw_20261019/_gates/$SLURM_JOB_ID
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
mkdir -p "$GATE_OUT"
cd "$GATE_OUT"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader || true
set +e
"$PYTHON" -u "$WT/scripts/v10_capacity/hw_gate.py" --out "$GATE_OUT" --device cuda
rc=$?
rm -rf "$GATE_OUT"/v10k_gate_*
exit "$rc"
