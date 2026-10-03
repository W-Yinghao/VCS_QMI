#!/usr/bin/env bash
#SBATCH --job-name=p123_gpu_smoke
#SBATCH --partition=A100,L40S,RTX6000PRO,H100
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P123 GPU smoke: full-size timing of one checkpoint (A-P3 C10 seed 0, epoch 800, 1 batch, all 11 counterfactual scorers) + C100 A-P3 epoch 800 (1 batch).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
J=${SLURM_JOB_ID:-local}; REP=$REPO_ROOT/reports/P123_GATE_SMOKE_$J; mkdir -p "$REP"; O=/home/infres/yinwang/CS_QMI/outputs/P123_GATE_SMOKE_$J
banner "P123 GPU smoke -> $REP"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" scripts/p123_grad.py --runs P107_AP3_views4_800ep_seed0,P107_AP3_c100_views4_800ep_seed0 --checkpoints 800 --grad-batches 1 --out-dir $O > "$REP/smoke.txt" 2>&1; rc=$?
grep -h "s  actual\|->" "$REP/smoke.txt"; cp $O/*.json "$REP/" 2>/dev/null
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
