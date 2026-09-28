#!/usr/bin/env bash
#SBATCH --job-name=estim_smoke
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/estim_smoke_%j.out
# Stand-alone CPU smoke of the P85 runner (subset of slurm/estim_v2_gate.sbatch): two tiny cells, all methods, then the aggregate.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=8
OUT=${OUT:-$OUTPUT_ROOT/P85_smoke_${SLURM_JOB_ID}}; mkdir -p "$OUT"
banner "P85 CPU smoke -> $OUT"
"$PYTHON" scripts/estim_make_units.py --out slurm/p85_units --chunks 8
"$PYTHON" scripts/estim_p85.py --units slurm/p85_units/smoke.txt --out "$OUT" --smoke --cpu; rc1=$?
"$PYTHON" scripts/estim_p85_aggregate.py --inputs "$OUT" --out "$OUT/aggregate"; rc2=$?
echo "rc=$rc1/$rc2 end=$(date -u +%FT%TZ)"; [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
