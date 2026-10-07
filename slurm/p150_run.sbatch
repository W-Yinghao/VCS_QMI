#!/usr/bin/env bash
#SBATCH --job-name=p150_run
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
#SBATCH --time=06:00:00
# P150 (FROZEN prereg): the crossed diagnostic, 20 draws, seed 601, on the feature store the gate regenerated.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
F=$OUTPUT_ROOT/P150_features; [ -f $F/P107_AP3_views4_800ep_seed1/features.pt ] || { echo "feature store missing"; exit 1; }
mkdir -p $OUTPUT_ROOT/P150_c0; "$PYTHON" scripts/p150_c0.py --features $F --run P107_AP3_views4_800ep_seed1 --out $OUTPUT_ROOT/P150_c0/c0 --repeats 20 --seed 601; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
