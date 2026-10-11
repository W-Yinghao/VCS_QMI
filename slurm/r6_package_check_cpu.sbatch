#!/usr/bin/env bash
#SBATCH --job-name=r6_package_check
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r6 package (downloaded; read before running): its math tests and reference examples, isolated (-I), inside the package copy.  Main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; rc=0
cd /home/infres/yinwang/CS_QMI/ssl_pilot/r6/package_r6_20261011 && { "$PYTHON" -I -m unittest discover -s tests -v || rc=1; "$PYTHON" -I -c "import sys, runpy; sys.path.insert(0, 'reference'); sys.argv = ['run_reference.py', '--out', 'validation/reference_examples_server.json']; runpy.run_path('reference/run_reference.py', run_name='__main__')" || rc=1; }   # -I keeps the script dir off sys.path: add only the reviewed reference/ dir
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
