#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101/logs/%x_%j.out
# R30 v1.1 branch unit (one MATRIX.csv row; RUN40 / RUN10 / RUN10_CONTROL of plan_v11/EXECUTION_PLAN.csv).
# Usage: sbatch --partition=<block partition> --exclude=node51,node53,node60 --time=... r30v11_unit.sbatch UNIT_ID
# tools/run_r30.py --stage unit --execute: gate PASS + frozen v1.1 protocol + code sha, bound parent (sha256 + fingerprint),
# registered coefficient for fit_static, block GPU, fork from the parent or resume this unit only, lock. On timeout without
# DONE.json the job resubmits itself (same name/args/partition), at most 3 attempts in total. No --dependency.
set -euo pipefail
UNIT=$1
source /home/infres/yinwang/CS_QMI/handoff/r30_pairwise_20261101/server.env
export PYTHONDONTWRITEBYTECODE=1
O=$R30_ROOT
OUT=$R30_OUT_BASE/runs/$UNIT
test -f $OUT/DONE.json && { echo "already DONE $OUT"; exit 0; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source "$R27_STAGE_SH"
mkdir -p $OUT
N=$(cat $OUT/ATTEMPTS 2>/dev/null || echo 0); N=$((N+1)); echo $N > $OUT/ATTEMPTS
echo "attempt $N unit $UNIT"
LIM=$(( $(squeue -j $SLURM_JOB_ID -h -o %L | awk -F'[-:]' '{ if (NF==4) print ($1*24+$2)*3600+$3*60+$4; else if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }') - 900 ))
set +e
timeout $LIM "$PYTHON_BIN" -u -B "$O/tools/run_r30.py" --stage unit --unit "$UNIT" --root $R30_ROOT --cells $R30_CELLS --code-dir $R30V11_CODE --data-path "$DP" --out-base $R30_OUT_BASE --python "$PYTHON_BIN" --gate-report $R30_GATE --bindings $R30_BINDINGS --plan $R30_PLAN --coefficients $R30_COEF --execute
rc=$?
set -e
if [ -f $OUT/DONE.json ]; then echo "R30_UNIT_DONE $UNIT"; exit 0; fi
if [ $rc -eq 124 ] && [ $N -lt 3 ]; then
  J=$(sbatch --parsable --job-name=$SLURM_JOB_NAME --partition=$SLURM_JOB_PARTITION --exclude=node51,node53,node60 --time=$(squeue -j $SLURM_JOB_ID -h -o %l) --nice=$(squeue -j $SLURM_JOB_ID -h -o %y) $O/sbatch/r30v11_unit.sbatch $UNIT)
  printf "%s\t%s\t%s\tcontinuation_of_%s\n" "$(date -u +%FT%TZ)" "$SLURM_JOB_NAME" "$J" "$SLURM_JOB_ID" >> $O/SUBMITTED.tsv
  echo "timeout: continuation submitted $J"; exit 0
fi
echo "R30_UNIT_FAILED rc=$rc attempt=$N"; exit 1
