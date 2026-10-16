#!/usr/bin/env bash
#SBATCH --job-name=v10_t01_t02_gpu
#SBATCH --partition=L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V10 phase 1 - one GPU job: (1) T01 on CUDA (package verify_torch.py --device cuda, design-tolerance core checks, real-host adapter
# acceptance fp64 + fp32 in both feature states, fp64 geometric gradient check, merge -> v10/T01 with the CPU parts); (2) T02
# timing (package benchmark_torch.py --full in float32 and float64, blocked-dense / permutation-variance extensions with the replay
# verified, encoder-inclusive train steps on 16 FIT cores, full registrations on 3 DEV cores with untrained fixed models - timing
# only) -> v10/T02/T02_TIMING_cuda.json + T02_LONG_cuda.csv, then BUDGET.json + UNIT_LEDGER.csv.
# L40S preferred (most V9 history); the header records the GPU.  To move this job while PENDING to other GPU types, update the
# own pending job (scontrol update JobId=<id> Partition=L40S,RTX6000PRO,H100); never submit a third V10 job.
#   slurm/v10/submit.sh slurm/v10/t01_t02_gpu.sbatch      (never plain sbatch: the header requires the submission pin)
set -o pipefail
R=/home/infres/yinwang/CS_QMI/ssl_pilot_v10
cd "$R" || exit 2
export PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python PYTHONUNBUFFERED=1 PYTHONPATH="$R/src:$R"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export V10_OUT_ROOT=/projects/EEG-foundation-model/yinghao/outputs_v10
source "$R/slurm/v10/v10_header.sh" || exit 2
v10_header gpu slurm/v10/t01_t02_gpu.sbatch || exit 3
mkdir -p "$V10_OUT_ROOT/T01" "$V10_OUT_ROOT/T02"
"$PYTHON" -m pytest tests/v10 -q -p no:cacheprovider 2>&1 | tail -4; echo "[v10] unit tests rc=${PIPESTATUS[0]} (recorded, not gating)"
"$PYTHON" -W ignore::UserWarning scripts/v10/t01_validate.py run --device cuda --parts verify_torch,core_tol,adapter,geometry; rc1=$?
"$PYTHON" -W ignore::UserWarning scripts/v10/t01_validate.py merge; rc_m=$?
"$PYTHON" -W ignore::UserWarning scripts/v10/t02_timing.py run --device cuda --parts head,head_extra,train,register; rc2=$?
"$PYTHON" -W ignore::UserWarning scripts/v10/t02_timing.py budget; rc3=$?
echo "[v10] t01_cuda_rc=$rc1 merge_rc=$rc_m t02_rc=$rc2 budget_rc=$rc3 end=$(date -u +%FT%TZ)"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ]
