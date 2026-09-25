#!/usr/bin/env bash
#SBATCH --job-name=vcs_dbg_tests
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/debug_tests_%j.out
# Fast CPU-only debug loop for the test-suite (no data needed).  Not part of the audited P0-P4 chain.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "DEBUG: CPU test run"
"$PYTHON" -m pytest -q tests/test_ssl_core.py -p no:cacheprovider 2>&1 | tail -3
rc1=${PIPESTATUS[0]}
"$PYTHON" -m pytest -q tests/test_integration.py -p no:cacheprovider 2>&1 | tail -60
rc2=${PIPESTATUS[0]}
echo "reference_rc=$rc1 integration_rc=$rc2 end=$(date -u +%FT%TZ)"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
