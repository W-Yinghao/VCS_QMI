#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nice=20
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_oracle_20261030/logs/%x_%j.out
# R27-O (ORACLE diagnostic) unit. Usage: r27o_unit.sbatch UNIT_ID   (UNIT_ID = <cell>__<arm>)
# tools/run_unit_o.py --execute (identity/hash checks, same-unit resume, lock, GPU_GATE.json all_pass + code sha required).
# Side watcher keep_states.py (R27-D, unchanged) hardlinks the epoch 10/20 full states. On timeout without DONE.json the
# job resubmits itself (same name/args), at most 3 attempts in total. No --dependency.
set -euo pipefail
UNIT=$1
source /home/infres/yinwang/CS_QMI/handoff/r27_oracle_20261030/server.env
O=$R27O_ROOT
CELL=${UNIT%%__*}; ARM=${UNIT##*__}
OUT=$R27O_OUTPUT/$CELL/$ARM
test -f $OUT/DONE.json && { echo "already DONE $OUT"; exit 0; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source "$R27_STAGE_SH"
mkdir -p $OUT
N=$(cat $OUT/ATTEMPTS 2>/dev/null || echo 0); N=$((N+1)); echo $N > $OUT/ATTEMPTS
echo "attempt $N unit $UNIT"
LIM=$(( $(squeue -j $SLURM_JOB_ID -h -o %L | awk -F'[-:]' '{ if (NF==4) print ($1*24+$2)*3600+$3*60+$4; else if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }') - 900 ))
"$PYTHON_BIN" -u -B $KEEP_STATES $OUT > $OUT/keep_states_${SLURM_JOB_ID}.log 2>&1 &
KP=$!
set +e
timeout $LIM "$PYTHON_BIN" -u -B "$O/tools/run_unit_o.py" --unit "$UNIT" --cells "$R27_ROOT/CELLS.json" \
  --lambda-file "$R27_ROOT/LAMBDA_flickr.json" --code-dir "$R27O_CODE" --data-path "$DP" --output-root "$R27O_OUTPUT" \
  --python "$PYTHON_BIN" --gate-report "$R27O_GATE" --execute
rc=$?
kill $KP 2>/dev/null || true
set -e
if [ -f $OUT/DONE.json ]; then echo "R27O_UNIT_DONE $UNIT"; exit 0; fi
if [ $rc -eq 124 ] && [ $N -lt 3 ]; then
  J=$(sbatch --parsable --job-name=$SLURM_JOB_NAME --time=$(squeue -j $SLURM_JOB_ID -h -o %l) --nice=$(squeue -j $SLURM_JOB_ID -h -o %y) $O/sbatch/r27o_unit.sbatch $UNIT)
  printf "%s\t%s\t%s\tcontinuation_of_%s\n" "$(date -u +%FT%TZ)" "$SLURM_JOB_NAME" "$J" "$SLURM_JOB_ID" >> $O/SUBMITTED.tsv
  echo "timeout: continuation submitted $J"; exit 0
fi
echo "R27O_UNIT_FAILED rc=$rc attempt=$N"; exit 1
