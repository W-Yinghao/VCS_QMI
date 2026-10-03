#!/usr/bin/env bash
#SBATCH --job-name=p124_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P124 CPU gate: tests (P124 + P119 / P110 regression), CPU smoke of the runner on one CIFAR-100 encoder (subset FIT/selection), aggregator on the smoke.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
G=reports/P124_GATE_${SLURM_JOB_ID:-local}; mkdir -p $G; rc=0
banner "P124 CPU gate -> $G"
"$PYTHON" -m pytest -q tests/test_p124.py 2>&1 | tee $G/pytest_test_p124.txt | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" -m pytest -q tests/test_p119.py tests/test_p110.py 2>&1 | tee $G/pytest_regression.txt | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" scripts/p124_gran.py --runs P107_AP3_c100_views4_800ep_seed0,P91_c100_simclr_views4_800ep_seed0 --out $G/smoke --smoke --cpu 2>&1 | tee $G/smoke_runner.txt | tail -8; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" scripts/p124_aggregate.py --in $G/smoke --out $G/smoke_aggregate 2>&1 | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
echo "{\"gate_rc\": $rc}" > $G/summary.json
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
