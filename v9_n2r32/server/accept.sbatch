#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-N2R32 step 4, GPU: student_runtime.py accept (seed 901, both residual arms, 50 real WeightedStudent updates
# each + one full DEV pass; no TEST).  Refuses unless RESIDUAL_DECISION.json is ELIGIBLE.  Writes
# c1/acceptance/<job>_<t>/C1_ACCEPTANCE.json and, on PASS, the gate file c1/C1_ACCEPTANCE.json.
set -u
source /home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2r32/v9_n2r32/server/env.sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
job_record start
"$PY_GPU" -I -B "$V9_PKG/student_runtime.py" accept --root "$N2R_ROOT" --repo "$V9_REPO" \
  --affine-root "$N2P_AFFINE_ROOT" --historical-root "$V9_HISTORY" --data "$V9_DATA" --vocab "$V9_VOCAB" \
  --tokens "$V9_TOKENS" --seed 901 --steps 50; rc=$?
job_record end "rc=$rc"; echo "end=$(date -u +%FT%TZ) rc=$rc"; exit $rc
