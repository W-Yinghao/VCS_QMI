#!/usr/bin/env bash
#SBATCH --job-name=p117_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P117 CPU gate: tests (P117 + P108 untouched), smokes of the three conditions, one full-size timing cell (C2_gauss_pad:16384:0, largest d / N).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
G=reports/P117_GATE; mkdir -p $G; rc=0; banner "P117 gate"
"$PYTHON" -m pytest -q tests/test_p117.py tests/test_p108.py > $G/pytest.txt 2>&1 || rc=1; tail -2 $G/pytest.txt
"$PYTHON" scripts/p117_cal.py --cpu --smoke --cells "C1_gauss_mid:4096:0;C2_gauss_pad:4096:0;C3_xor:4096:0" --out-dir /home/infres/yinwang/CS_QMI/outputs/P117_GATE_smoke > $G/smoke.txt 2>&1 || rc=1; tail -4 $G/smoke.txt
/usr/bin/env bash -c "time '$PYTHON' scripts/p117_cal.py --cpu --cells C2_gauss_pad:16384:0 --out-dir /home/infres/yinwang/CS_QMI/outputs/P117_GATE_timing" > $G/timing_cpu.txt 2>&1 || rc=1; tail -5 $G/timing_cpu.txt
echo "gate_rc=$rc exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
