#!/usr/bin/env bash
#SBATCH --job-name=p121_theory
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P121 (v6 V6-THEORY): tests, then the full exact / finite checks -> reports/P121_theory_checks.json, reports/P121_gram_fit.csv, reports/P121_conditions.md.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P121 theory"; rc=0
"$PYTHON" -m pytest -q tests/test_p121.py > reports/P121_GATE/pytest.txt 2>&1 || rc=1; tail -2 reports/P121_GATE/pytest.txt
[ $rc -eq 0 ] && { "$PYTHON" scripts/p121_theory.py --out-prefix reports/P121 || rc=$?; }
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
