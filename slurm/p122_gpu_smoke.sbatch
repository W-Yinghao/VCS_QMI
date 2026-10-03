#!/usr/bin/env bash
#SBATCH --job-name=p122_gpu_smoke
#SBATCH --partition=A100,L40S,RTX6000PRO,H100
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P122 GPU smoke / timing: (1) FULL-size run of one encoder x one law (A-P3 CIFAR-10, common-standard) to measure the real per-(encoder, law) cost
# (augmented images and features land in the production cache and are reused by the full job); (2) smoke-size CIFAR-100 path on GPU.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
G=$REPO_ROOT/reports/P122_GATE_SMOKE_${SLURM_JOB_ID:-local}; mkdir -p "$G"; rc=0
banner "P122 GPU smoke -> $G"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" scripts/p122_evidence.py --dataset cifar10 --runs P107_AP3_views4_800ep_seed0 --laws standard --out "$G/full_c10" || rc=1
"$PYTHON" scripts/p122_evidence.py --dataset cifar100 --runs P107_AP3_c100_views4_800ep_seed0 --laws standard --out "$G/smoke_c100" \
   --cache /home/infres/yinwang/CS_QMI/outputs/P122_GATE_cache --smoke || rc=1
echo "smoke_rc=$rc end=$(date -u +%FT%TZ)"; exit $rc
