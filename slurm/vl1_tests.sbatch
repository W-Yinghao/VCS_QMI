#!/usr/bin/env bash
#SBATCH --job-name=vl1_tests
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
"$PYTHON" -m pytest -q tests/test_vl1.py 2>&1 | tail -3; rc=${PIPESTATUS[0]}; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
