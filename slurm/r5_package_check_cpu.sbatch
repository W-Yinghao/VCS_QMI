#!/usr/bin/env bash
#SBATCH --job-name=r5_package_check
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r5 package (downloaded, read before running): its reference tests and manifest check, isolated (-I); then our r5 / R4 loss tests.  Main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; rc=0
cd /home/infres/yinwang/CS_QMI/ssl_pilot/r5/package_r5_20261011 && { "$PYTHON" -I -m unittest discover -s tests -v || rc=1; "$PYTHON" -I reference/check_package.py || rc=1; }
cd /home/infres/yinwang/CS_QMI/ssl_pilot && { "$PYTHON" -m unittest tests/test_r5_distribution_softmax.py tests/test_r4_vl2_nu.py tests/test_r4_vl1_laws.py -v || rc=1; }
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
