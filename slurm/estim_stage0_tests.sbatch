#!/usr/bin/env bash
#SBATCH --job-name=estim_pkg_stage0_tests
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/estim_pkg_stage0_tests_%j.out
# Estimator package v1, stage 0 (§5) test gate + the existing reference-core suite; then a CPU smoke of the stage-1/2 probe.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "estimator package: stage-0 tests + probe smoke"
"$PYTHON" -m pytest -q tests/test_vcs_estim_stage0.py -p no:cacheprovider 2>&1 | tail -40; rc1=${PIPESTATUS[0]}
"$PYTHON" -m pytest -q tests/test_ssl_core.py -p no:cacheprovider 2>&1 | tail -3; rc2=${PIPESTATUS[0]}
PYTHONPATH=src "$PYTHON" -m vcs_estim.run --I 4 --smoke --cpu --out "${SMOKE_OUT:-/tmp/estim_smoke_$SLURM_JOB_ID.json}" 2>&1 | tail -40; rc3=${PIPESTATUS[0]}
echo "stage0_rc=$rc1 reference_rc=$rc2 smoke_rc=$rc3 end=$(date -u +%FT%TZ)"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 0 ]
