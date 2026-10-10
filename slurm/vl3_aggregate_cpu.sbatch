#!/usr/bin/env bash
#SBATCH --job-name=vl3_aggregate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL3 readings: B/16 + Large (add. 1) and the 20-epoch schedule check (add. 2); main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; rc=0
echo "== VL3 (B/16 + Large)"; "$PYTHON" scripts/vl3_aggregate.py || rc=1
echo "== VL3 add. 2 / 4 (20 vs 10 epochs; paired)"; "$PYTHON" scripts/vl3_add2_aggregate.py || rc=1
echo "== VL3 add. 4 (20-epoch runs, all seeds)"; "$PYTHON" scripts/vl3_aggregate.py --suffix _e20 || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
