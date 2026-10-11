#!/usr/bin/env bash
#SBATCH --job-name=r6gain_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r6 GAIN-P1 on CPU (scripts/r6_gain_pilot.py; first arg "tests" runs the unit tests, otherwise args are passed through).  Main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
if [ "$1" = "tests" ]; then "$PYTHON" -m unittest tests/test_r6_gain_pilot.py -v; rc=$?; else "$PYTHON" scripts/r6_gain_pilot.py "$@"; rc=$?; fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
