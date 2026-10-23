#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-N2R32 step 1, CPU: the package's single CPU check entry (v9/tests/run_cpu_checks.py) with --repo, so the
# original weight reader test also runs.  Synthetic data only.  usage: sbatch [opts] cpu_checks.sbatch [OUT_NAME]
set -u
source /home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2r32/v9_n2r32/server/env.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
name=${1:-CPU_CHECKS}
job_record start
"$PY_CPU" -B "$V9_PKG/tests/run_cpu_checks.py" --repo "$V9_REPO" --out "$N2R_ROOT/validation/$name.json"; rc=$?
job_record end "rc=$rc"; echo "rc=$rc"; exit $rc
