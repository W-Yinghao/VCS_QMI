#!/usr/bin/env bash
#SBATCH --job-name=n2m_fixscore
#SBATCH --partition=cpu-high
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# Package step 5, part 2: EVAL_ONLY reweighting of each seed's fixed-student loss export (score-fixed-losses).
# Runs after the three export jobs (n2m_loss901/902/903); CPU only; nothing here feeds training or control generation.
set -u
source /home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2m/v9_n2m/server/env.sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
cd "$N2M_PACKAGE" || exit 3
rc_all=0
for s in 901 902 903; do
  "$N2M_PYTHON" "$N2M_PACKAGE/run_n2m.py" score-fixed-losses --experiment "$N2M_OUT/experiment.json" --seed "$s" \
    --labels "$N2M_OLD_OUTPUT/rho0.4/s$s/eval_only/pairs.eval_only.json" \
    --loss-csv "$N2M_OUT/diagnostics/s$s/fixed_losses.csv" --out "$N2M_OUT/analysis/fixed_loss/s$s"; rc=$?
  echo "fixscore_s${s}_rc=$rc"; [ $rc -ne 0 ] && rc_all=$rc
done
echo "end=$(date -u +%FT%TZ)"; exit $rc_all
