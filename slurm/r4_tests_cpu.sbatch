#!/usr/bin/env bash
#SBATCH --job-name=r4_tests
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R4 unit tests (args: test module paths relative to ssl_pilot); main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; cd /home/infres/yinwang/CS_QMI/ssl_pilot
"$PYTHON" -m unittest "$@" -v; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
