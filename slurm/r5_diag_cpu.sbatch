#!/usr/bin/env bash
#SBATCH --job-name=r5_diag
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r5 analyses (scripts/r5_fusion.py, scripts/r5_diagnostics.py) and their tests; first arg selects: tests | fusion | diagnostics; the rest passed through.  Main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot; what="$1"; shift
case "$what" in
  tests) "$PYTHON" -m unittest tests/test_r5_diagnostics.py -v; rc=$?;;
  fusion) "$PYTHON" scripts/r5_fusion.py "$@"; rc=$?;;
  diagnostics) "$PYTHON" scripts/r5_diagnostics.py "$@"; rc=$?;;
  *) echo "usage: sbatch slurm/r5_diag_cpu.sbatch tests|fusion|diagnostics [args]"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
