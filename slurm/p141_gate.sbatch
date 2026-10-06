#!/usr/bin/env bash
#SBATCH --job-name=p141_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P141 CPU gate: tests (P141 + P124 regression), CPU smoke of the runner on two CIFAR-100 encoders (subset FIT / selection, all four sites,
# default readouts), aggregator on the smoke results.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
G=reports/P141_GATE_${SLURM_JOB_ID:-local}; mkdir -p $G; rc=0
banner "P141 CPU gate -> $G"
"$PYTHON" -m pytest -q tests/test_p141.py 2>&1 | tee $G/pytest_test_p141.txt | tail -3; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" -m pytest -q tests/test_p124.py 2>&1 | tee $G/pytest_regression_p124.txt | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
t0=$(date +%s); "$PYTHON" scripts/p141_layers.py --runs P107_AP3_c100_views4_800ep_seed0,P91_c100_simclr_views4_800ep_seed0 --out $G/smoke --smoke --cpu 2>&1 | tee $G/smoke_runner.txt | tail -12; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
echo "smoke wall seconds: $(( $(date +%s) - t0 ))" | tee -a $G/smoke_runner.txt
"$PYTHON" scripts/p141_aggregate.py --in $G/smoke --out $G/smoke_aggregate --include-smoke 2>&1 | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
echo "{\"gate_rc\": $rc}" > $G/summary.json
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
