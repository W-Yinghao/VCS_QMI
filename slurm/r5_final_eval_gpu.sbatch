#!/usr/bin/env bash
#SBATCH --job-name=r5fe_gpu
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r5 final evaluation, GPU stages of scripts/r5_final_eval.py: proposals | rank (--shard <name>).  First argument = stage, the rest passed through.
# The driver submits proposals with --partition=RTX6000PRO (the GPU type of the reported DEV / CAL proposals) and --time=00:45:00; --exclude stays.
# Time limits are kept short (~3x the measured cost: rank shard <= ~50 min on L40S) so that the jobs can backfill.
E=/projects/EEG-foundation-model/yinghao/envs/vl_baselines
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd /home/infres/yinwang/CS_QMI/ssl_pilot || exit 2; what="${1:-}"; shift
echo "stage=$what host=$(hostname) job=${SLURM_JOB_ID:-NA} $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) commit=$(git rev-parse --short HEAD 2>/dev/null) dirty=$(git status --porcelain -- scripts/r5_final_eval.py slurm/r5_final_eval_cpu.sbatch slurm/r5_final_eval_gpu.sbatch 2>/dev/null | wc -l) start=$(date -u +%FT%TZ)"
case "$what" in
  proposals|rank) "$E/bin/python" -I scripts/r5_final_eval.py "$what" "$@"; rc=$?;;
  *) echo "usage: sbatch slurm/r5_final_eval_gpu.sbatch proposals|rank [args]"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
