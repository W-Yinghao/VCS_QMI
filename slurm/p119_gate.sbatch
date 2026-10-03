#!/usr/bin/env bash
#SBATCH --job-name=p119_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P119 CPU gate: tests (P119 + P110 regression), aggregator on the committed P110 data (seed-paired label efficiency), CPU smoke of every runner
# module on one encoder (fpcheck on CPU is a code-path check only: the cache was extracted on GPU, so CPU-vs-cache mixes device and precision).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
G=reports/P119_GATE_${SLURM_JOB_ID:-local}; mkdir -p $G; rc=0
banner "P119 CPU gate -> $G"
"$PYTHON" -m pytest -q tests/test_p119.py 2>&1 | tee $G/pytest_test_p119.txt | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" -m pytest -q tests/test_p110.py 2>&1 | tee $G/pytest_test_p110.txt | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" scripts/p119_aggregate.py --in $G/none --out $G/label_efficiency_paired_p110 || rc=1
"$PYTHON" scripts/p119_v.py --runs P41_simclr_views4_800ep_seed0 --modules transfer,c10extra --out $G/smoke --smoke --cpu 2>&1 | tee $G/smoke_runner.txt | tail -4; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" scripts/p119_v.py --runs P41_simclr_views4_800ep_seed0 --modules transfer --out $G/smoke_fp32 --smoke --cpu --fp32 2>&1 | tail -2; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
"$PYTHON" scripts/p119_v.py --runs P41_simclr_views4_800ep_seed0 --modules fpcheck --out $G/smoke --cpu 2>&1 | tee $G/smoke_fpcheck_cpu.txt | tail -3; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
echo "{\"gate_rc\": $rc}" > $G/summary.json
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
