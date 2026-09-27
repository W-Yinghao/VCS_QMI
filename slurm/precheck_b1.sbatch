#!/usr/bin/env bash
#SBATCH --job-name=precheck_b1
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/precheck_b1_%j.out
# Pre-check B question 1 (closed-form vs neural critic) on the CPU partition. RUNS = run_id:ckpt items separated by ";" (sbatch --export splits on commas); REPORT = output prefix.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
: "${RUNS:?}"; : "${REPORT:?}"
banner "pre-check B1: ${RUNS}"
"$PYTHON" scripts/precheck_b1_closedform.py --runs "$RUNS" --out "$REPORT"; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
