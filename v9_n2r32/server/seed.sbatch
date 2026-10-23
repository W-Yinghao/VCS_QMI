#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9-N2R32 step 5, GPU: student_runtime.py run for one seed -- both residual arms, 40 epochs each, sequentially in
# the protocol order (901/903 VCS->BCE, 902 BCE->VCS) on this one GPU.  Requires c1/C1_ACCEPTANCE.json PASS.
# Submitted on RTX6000PRO only (see SERVER_DELTA.md: backend identity with the affine C1 units, both arms in 24 h).
# After TIMEOUT/failure resubmit the identical command, again on RTX6000PRO: a finished arm is
# skipped, an unfinished arm resumes from its last complete epoch; exit 76 = GPU/driver/software differ from an
# earlier attempt of this seed.  usage: sbatch [opts] seed.sbatch SEED
set -u
seed=${1:?usage: seed.sbatch SEED}
source /home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2r32/v9_n2r32/server/env.sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
trap 'echo "SIGTERM $(date -u +%FT%TZ)"' TERM      # let the runtime write its INTERRUPTED records, then log the end
job_record start "seed=$seed"
"$PY_GPU" -I -B "$V9_PKG/student_runtime.py" run --root "$N2R_ROOT" --repo "$V9_REPO" \
  --affine-root "$N2P_AFFINE_ROOT" --historical-root "$V9_HISTORY" --data "$V9_DATA" --vocab "$V9_VOCAB" \
  --tokens "$V9_TOKENS" --seed "$seed"; rc=$?
job_record end "seed=$seed rc=$rc"; echo "end=$(date -u +%FT%TZ) rc=$rc"; exit $rc
