#!/usr/bin/env bash
#SBATCH --job-name=v81_d02_d04a
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/v81_d02_d04a_%j.out
# V8.1 D02 / D03 / D04a (CPU, zero training): normalise -> package diagnose.py -> D02 / D03 / D04a summaries.
# Every step skips outputs that already exist (resumable); STEP=<name> runs a single step.
set -u
PYTHON=/home/infres/yinwang/CS_QMI/env/bin/python
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd /home/infres/yinwang/CS_QMI/ssl_pilot_v81 || exit 2
echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
STEP=${STEP:-all}
run() {
  local name=$1; shift
  if [ "$STEP" != all ] && [ "$STEP" != "$name" ]; then return 0; fi
  echo "== $name start=$(date -u +%FT%TZ)"
  "$PYTHON" "$@"; local rc=$?
  echo "== $name exit=$rc end=$(date -u +%FT%TZ)"
  [ $rc -eq 0 ] || exit $rc
}
run normalize scripts/v81/d02_normalize.py --what all
run diagnose scripts/v81/d02_agreement.py run
run d02 scripts/v81/d02_agreement.py summarise
run d03 scripts/v81/d03_scale_selection.py
run d04a scripts/v81/d04a_distribution.py
echo "all done end=$(date -u +%FT%TZ)"
