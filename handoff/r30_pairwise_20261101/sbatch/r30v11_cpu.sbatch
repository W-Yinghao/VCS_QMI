#!/bin/bash
#SBATCH --job-name=r30v11_cpu
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101/logs/%x_%j.out
# R30 v1.1 CPU checks (tools/cpu_check_v11.py, synthetic) + the v1.1 package's own tools/run_math_checks.py (package CPU
# reference check, recorded separately; not a production gate).
set -euo pipefail
source /home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101/server.env
export PYTHONDONTWRITEBYTECODE=1
cd $R30_ROOT
$PYTHON_BIN -B $R30V11_PKG/tools/run_math_checks.py --output $R30_ROOT/v11_package_checks/math_checks_${SLURM_JOB_ID}.json
$PYTHON_BIN -B tools/cpu_check_v11.py $R30V11_CODE $R30_ROOT/reference $R30V11_PKG $R30_ROOT/CPU_CHECK_V11_${SLURM_JOB_ID}.json
cp $R30_ROOT/CPU_CHECK_V11_${SLURM_JOB_ID}.json $R30_ROOT/CPU_CHECK_V11.json
echo CPU_DONE
