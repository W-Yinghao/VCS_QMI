#!/usr/bin/env bash
#SBATCH --job-name=r6reg2_tests
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R6 REG-P2 unit tests together with REG-P1's (the module P2 imports).  Main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
"$PYTHON" -m unittest tests/test_r6_reg_p2.py tests/test_r6_reg_pilot.py -v; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
