#!/usr/bin/env bash
#SBATCH --job-name=r6crema_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r6 GAIN-P1 CREMA-D on CPU: first arg "tests" runs the unit tests, otherwise args go to scripts/r6_gain_crema.py (e.g. aggregate --mode cv5).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
export PYTHON=/projects/EEG-foundation-model/yinghao/envs/r6_av/bin/python OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
if [ "$1" = "tests" ]; then "$PYTHON" -m unittest tests/test_r6_gain_crema.py -v; rc=$?; else "$PYTHON" scripts/r6_gain_crema.py "$@"; rc=$?; fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
