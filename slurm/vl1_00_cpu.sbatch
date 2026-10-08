#!/usr/bin/env bash
#SBATCH --job-name=vl1_00_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL1-00 / VL1-01 (CPU): the package's own reference tests (run in its unpacked copy), the ported tests (tests/test_vl1.py), and the
# RefCOCOg data audit (reports/VL1/VL1_DATA_AUDIT.json, roles, box-check sheets).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
PKG=$1; G=reports/VL1/gate_cpu_${SLURM_JOB_ID:-local}; mkdir -p $G; rc=0
( cd $PKG && PYTHONPATH=. "$PYTHON" -m unittest discover -s tests -v ) > $G/package_unittest.txt 2>&1; tail -3 $G/package_unittest.txt; grep -q "^OK" $G/package_unittest.txt || rc=1
( cd $PKG && PYTHONPATH=. "$PYTHON" reference/export_math_checks.py ) > $G/package_math_checks.txt 2>&1 || rc=1; tail -3 $G/package_math_checks.txt
"$PYTHON" -m pytest -q tests/test_vl1.py 2>&1 | tee $G/pytest_vl1.txt | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" scripts/vl1_00_audit.py 2>&1 | tee $G/audit.txt | tail -12; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
