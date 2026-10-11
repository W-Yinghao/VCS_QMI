#!/usr/bin/env bash
#SBATCH --job-name=r6gain_check_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r6 GAIN-P1 checks after the audit / aggregate / memory patch: unit tests, standardised audio duplicate audit (full roles), CREMA-D actor
# roles (file names only), one patched CPU smoke run (brier, seed 1), smoke aggregate.  Main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot; rc=0
"$PYTHON" scripts/r6_gain_pilot.py roles --dataset cremad || rc=1
"$PYTHON" -m unittest tests/test_r6_gain_pilot.py -v || rc=1
"$PYTHON" scripts/r6_gain_pilot.py audit --dataset avmnist || rc=1
"$PYTHON" scripts/r6_gain_pilot.py run --dataset avmnist --family brier --seed 1 --smoke --cpu || rc=1
"$PYTHON" scripts/r6_gain_pilot.py aggregate --dataset avmnist --smoke || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
