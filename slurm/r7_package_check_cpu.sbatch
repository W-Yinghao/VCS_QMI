#!/usr/bin/env bash
#SBATCH --job-name=r7_package_check
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r7 package (downloaded; read before running): its math tests, isolated (-I), inside the package copy (the tests add reference/ themselves).  Main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; rc=0
cd /home/infres/yinwang/CS_QMI/ssl_pilot/r7/package_r7_20261011 && { "$PYTHON" -I -m unittest discover -s tests -v || rc=1; }
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
