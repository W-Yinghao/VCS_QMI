#!/usr/bin/env bash
#SBATCH --job-name=n2pc_acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V9 N2P C1 technical acceptance, GPU, no TEST: seed 901, both rich arms in protocol order, 50 real updates each
# through the original WeightedStudent plus one full-DEV pass.  Refuses unless PILOT_DECISION.json is
# ELIGIBLE_PENDING_TECHNICAL_ACCEPTANCE.  Writes c1/acceptance/<job>_<t>/C1_ACCEPTANCE.json and, on PASS, the gate
# file c1/C1_ACCEPTANCE.json.  Partition/qos/exclude/nice come from the sbatch line.
set -u
source /home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2p_c/v9_n2p/c_server/env.sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) job=${SLURM_JOB_ID:-} start=$(date -u +%FT%TZ)"
"$N2PC_PY_C1" -I -B "$N2PC_SERVER/c1_student.py" accept --root "$N2PC_ROOT" --seed 901 --steps 50; rc=$?
mkdir -p "$N2PC_RECORDS/c1"; cp "$N2PC_ROOT/c1/C1_ACCEPTANCE.json" "$N2PC_RECORDS/c1/" 2>/dev/null
cp "$N2PC_ROOT"/c1/acceptance/"${SLURM_JOB_ID:-nojob}"_*/C1_ACCEPTANCE.json "$N2PC_RECORDS/c1/C1_ACCEPTANCE_${SLURM_JOB_ID:-nojob}.json" 2>/dev/null
cp "$N2PC_ROOT"/c1/traces/accept_s901_${SLURM_JOB_ID:-nojob}_*.json "$N2PC_RECORDS/c1/" 2>/dev/null
echo "end=$(date -u +%FT%TZ) rc=$rc"; exit $rc
