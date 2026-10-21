#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9 N2P C1 seed pair, GPU: both rich arms of SEED, 40 epochs each, sequentially in the protocol order
# (901 VCS->BCE, 902 BCE->VCS, 903 VCS->BCE) on this one GPU.  Requires PILOT_DECISION ELIGIBLE and
# c1/C1_ACCEPTANCE.json PASS.  Submit with ONE partition per seed (--job-name=n2pc_s<SEED> --time=...).  After a
# time limit or failure resubmit the identical line: a finished arm is skipped, an unfinished arm resumes from
# its last complete epoch; exit 76 means the allocation's GPU/software differs from the pair's first attempt.
# usage: sbatch [options] c1_seed.sbatch SEED
set -u
seed=${1:?usage: c1_seed.sbatch SEED}
source /home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2p_c/v9_n2p/c_server/env.sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) job=${SLURM_JOB_ID:-} partition=${SLURM_JOB_PARTITION:-} seed=$seed start=$(date -u +%FT%TZ)"
"$N2PC_PY_C1" -I -B "$N2PC_SERVER/c1_student.py" run --root "$N2PC_ROOT" --seed "$seed"; rc=$?
mkdir -p "$N2PC_RECORDS/c1"
for d in "$N2PC_ROOT"/c1/runs/s"$seed"__*; do
  [ -d "$d" ] || continue; u=$(basename "$d")
  cp "$d/attempts.jsonl" "$N2PC_RECORDS/c1/${u}_attempts.jsonl" 2>/dev/null
  cp "$d/unit_result.json" "$N2PC_RECORDS/c1/${u}_unit_result.json" 2>/dev/null
done
cp "$N2PC_ROOT"/c1/traces/run_s${seed}_${SLURM_JOB_ID:-nojob}_*.json "$N2PC_RECORDS/c1/" 2>/dev/null
echo "end=$(date -u +%FT%TZ) rc=$rc"; exit $rc
