#!/usr/bin/env bash
#SBATCH --job-name=r6gain_smoke_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r6 GAIN-P1 CPU smoke: unit tests, role split (full), audio duplicate audit (full roles), smoke runs (both families, seed 0, subsets), smoke aggregate.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot; rc=0
"$PYTHON" -m unittest tests/test_r6_gain_pilot.py -v || rc=1
"$PYTHON" scripts/r6_gain_pilot.py roles --dataset avmnist || rc=1
"$PYTHON" scripts/r6_gain_pilot.py audit --dataset avmnist || rc=1
for f in brier ce; do "$PYTHON" scripts/r6_gain_pilot.py run --dataset avmnist --family $f --seed 0 --smoke --cpu || rc=1; done
"$PYTHON" scripts/r6_gain_pilot.py aggregate --dataset avmnist --smoke || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
