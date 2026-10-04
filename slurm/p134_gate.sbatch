#!/usr/bin/env bash
#SBATCH --job-name=p134_gate
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p134_%x_%j.out
# P134 CPU gate: unit tests + a CPU smoke of three tiny cells + the aggregator on the smoke output.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P134 gate"
OUTD=/home/infres/yinwang/CS_QMI/outputs/P134_GATE_${SLURM_JOB_ID}
"$PYTHON" -m pytest -q tests/test_vcs_estim_p134.py; r1=$?
"$PYTHON" scripts/p134_r1.py --units slurm/p134_units/smoke.txt --out "$OUTD" --smoke --cpu --N 256 --updates 40; r2=$?
"$PYTHON" scripts/p134_aggregate.py --p134 "$OUTD" --p85 /home/infres/yinwang/CS_QMI/outputs/P85_estim_benchmark --out "$OUTD/agg"; r3=$?
echo "pytest=$r1 smoke=$r2 aggregate=$r3"; rc=$(( r1 | r2 | r3 )); echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
