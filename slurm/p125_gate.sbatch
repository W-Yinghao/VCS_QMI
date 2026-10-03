#!/usr/bin/env bash
#SBATCH --job-name=p125_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P125 CPU gate: P125 tests + P108 / P117 suites (untouched), smoke cells (orig + rot), aggregate.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
G=reports/P125_GATE_${SLURM_JOB_ID:-local}; mkdir -p $G; rc=0; banner "P125 gate"
"$PYTHON" -m pytest -q tests/test_p125.py tests/test_p108.py tests/test_p117.py > $G/pytest.txt 2>&1 || rc=1; tail -2 $G/pytest.txt
"$PYTHON" scripts/p125_struct.py --cpu --smoke --cells "C1_gauss_mid:4096:0:orig;C2_gauss_pad:4096:0:orig;C3_xor:4096:0:orig;C1_gauss_mid:4096:0:rot" --out-dir /home/infres/yinwang/CS_QMI/outputs/P125_GATE_smoke > $G/smoke.txt 2>&1 || rc=1; tail -5 $G/smoke.txt
"$PYTHON" scripts/p125_aggregate.py --in-dir /home/infres/yinwang/CS_QMI/outputs/P125_GATE_smoke --out $G/smoke_aggregate > $G/aggregate.txt 2>&1 || rc=1; tail -1 $G/aggregate.txt
echo "gate_rc=$rc exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
