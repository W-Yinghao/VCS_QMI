#!/usr/bin/env bash
#SBATCH --job-name=n2m_prepare
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Freeze the N2M 6-9-run matrix (package step 3): census of the 33 inherited N2F units with checkpoint SHA, Constant-U /
# Shuffle-VCS generation + read-back, reference decisions (auto), experiment.json / EXPERIMENT.sha256 / UNIT_PLAN.csv.
# Requires N2F 33/33 and OUT/SERVER_VERIFICATION.json status PASS.  CPU only; no eval_only labels; no training.
# Job name has no v9 prefix on purpose: the N2F closeout controllers 1047107 / 1046165 wait while any job named v9* is live.
set -u
source /home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/server/env.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "host=$(hostname) start=$(date -u +%FT%TZ) job=${SLURM_JOB_ID:-}"
cd "$N2M_PACKAGE" || exit 3
"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" verify-package > /dev/null || exit 4
"$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" prepare --base "$N2M_BASE" --out "$N2M_OUT" --mode auto; rc=$?
echo "prepare_rc=$rc end=$(date -u +%FT%TZ)"; exit $rc
