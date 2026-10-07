#!/usr/bin/env bash
#SBATCH --job-name=p150_gate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P150 gate: regenerate the P143 feature store for the one encoder (deterministic extraction; the P143 stage-0 gate already validated this path),
# then a smoke of scripts/p150_c0.py (2 draws, n 200).  Full-size feature store kept under outputs/P150_features for the frozen run.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
G=reports/P150_GATE_${SLURM_JOB_ID:-local}; mkdir -p $G; rc=0; F=$OUTPUT_ROOT/P150_features
[ -f $F/P107_AP3_views4_800ep_seed1/features.pt ] || { "$PYTHON" scripts/p109_i1_features.py --runs P107_AP3_views4_800ep_seed1 --out $F --fp32-maps 2>&1 | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1; }
"$PYTHON" scripts/p150_c0.py --features $F --run P107_AP3_views4_800ep_seed1 --out $OUTPUT_ROOT/P150_smoke/c0 --smoke 2>&1 | tail -6; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
echo "{\"gate_rc\": $rc}" > $G/summary.json; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
