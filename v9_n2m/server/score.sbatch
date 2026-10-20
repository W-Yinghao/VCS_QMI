#!/usr/bin/env bash
#SBATCH --job-name=n2m_score
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Package step 5, part 1: offline EVAL_ONLY posterior / mass diagnostics per seed (score).  This is the only step that
# reads the mechanism labels pairs.eval_only.json; it runs after prepare froze the controls; it trains nothing and its
# output never feeds training, calibration choice or control generation.  usage: sbatch score.sbatch [SEED ...]
set -u
source /home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/server/env.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "host=$(hostname) start=$(date -u +%FT%TZ) job=${SLURM_JOB_ID:-}"
cd "$N2M_PACKAGE" || exit 3
rc_all=0
[ $# -eq 0 ] && set -- 901 902 903
for s in "$@"; do
  "$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" score --experiment "$N2M_OUT/experiment.json" --seed "$s" \
    --labels "$N2M_OLD_OUTPUT/rho0.4/s$s/eval_only/pairs.eval_only.json" --out "$N2M_OUT/analysis/offline/s$s"; rc=$?
  echo "score_s${s}_rc=$rc"; [ $rc -ne 0 ] && rc_all=$rc
done
echo "end=$(date -u +%FT%TZ)"; exit $rc_all
