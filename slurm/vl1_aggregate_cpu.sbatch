#!/usr/bin/env bash
#SBATCH --job-name=vl1_aggregate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Frozen-reading aggregations (VL1-10 / 11, VL1-12, Table B on DEV, P145 addendum 2); main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; rc=0
echo "== VL1-10 / 11"; "$PYTHON" scripts/vl1_10_aggregate.py || rc=1
echo "== VL1-12 table"; "$PYTHON" scripts/vl1_10_aggregate.py --dir outputs/VL1_12 --out reports/VL1/VL1_12_table.json --ns all --routes vcs js softmax rff || rc=1
echo "== VL1-12 reading"; "$PYTHON" scripts/vl1_12_compare.py || rc=1
echo "== Table B (DEV)"; "$PYTHON" scripts/vl1_tableB_aggregate.py || rc=1
echo "== P145 add. 2"; "$PYTHON" scripts/p145a2_aggregate.py || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
