#!/bin/bash
# R33 benchmark baseline unit.  r33_unit.sh KIND CELL [formal|smoke]
#   KIND prefix   : native IN2R (code_h0 = byte copy of R25 code) warmup 5 + co-train epoch 0 -> prefix/<cell>/prefix_state.pt
#        H0       : native IN2R full unit from the cell prefix (code_h0/r25_run.py, R25 code-version config; = R26 H0)
#        a1_iface : IN2R + A1 interface from the same prefix (code_a1 = byte copy of R27-A1X code_a1x, r27_run.py --arm a1_iface --lam 0)
#   CELL from CELLS.json (c60/f80/c20/c80/f20 x s1001-3).
# Resume: preemption -> Slurm requeue reruns this script; a unit resumes from its own state_last.pt (else from the prefix);
# a prefix restarts from scratch unless prefix_state.pt exists (atomic save). Wall-clock: python runs under `timeout`;
# on timeout the job resubmits itself (<8 attempts; the feeder also resubmits unfinished units with no live job).
# smoke: prefix with --max_iters 20, each arm 2 co-train epochs x 20 iters, then a resume run of 1 more epoch.
set -euo pipefail
KIND=$1; CELL=$2; MODE=${3:-formal}
D=/home/infres/yinwang/CS_QMI/handoff/r33_baselines_20261102
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r33_baselines_20261102
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
eval "$($PY -B - "$D/CELLS.json" "$CELL" "$KIND" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))[sys.argv[2]]
assert sys.argv[3] == 'prefix' or sys.argv[3] in c['arms'], (sys.argv[3], c['arms'])
for k in ('data_name', 'noise_ratio', 'seed', 'noise_file', 'prefix'):
    print(f"{k.upper()}={c[k]}")
print("ARMS='" + ' '.join(c['arms']) + "'")
PYEOF
)"
if [ "$MODE" = smoke ]; then BASE=$O/smoke/$CELL; PFX=$BASE/prefix/prefix_state.pt; SX="--max_iters 20 --log_step 10"
elif [ "$MODE" = formal ]; then BASE=$O; PFX=$PREFIX; SX=""
else echo "bad mode $MODE"; exit 2; fi
case $KIND in
  prefix)   OUT=$(dirname $PFX); CODE=$D/code_h0;;
  H0)       OUT=$BASE/runs/$CELL/H0; [ $MODE = smoke ] && OUT=$BASE/H0; CODE=$D/code_h0;;
  a1_iface) OUT=$BASE/runs/$CELL/a1_iface; [ $MODE = smoke ] && OUT=$BASE/a1_iface; CODE=$D/code_a1;;
  *) echo "bad kind $KIND"; exit 2;;
esac
mkdir -p $OUT

# delete the cell prefix once every arm of the cell has its own state_last.pt (formal only)
prefix_gc() {
  [ "$MODE" = formal ] && [ -f $PREFIX ] || return 0
  for a in $ARMS; do [ -f $O/runs/$CELL/$a/state_last.pt ] || return 0; done
  printf '{"deleted_utc": "%s", "path": "%s", "bytes": %s, "reason": "all arms (%s) have their own state_last.pt", "by_job": "%s"}\n' \
    "$(date -u +%FT%TZ)" "$PREFIX" "$(stat -c %s $PREFIX)" "$ARMS" "${SLURM_JOB_ID:-}" > $(dirname $PREFIX)/PREFIX_DELETED.json
  rm -f $PREFIX; echo "prefix deleted: $PREFIX"
}

if [ $KIND = prefix ]; then
  if [ -f $OUT/PREFIX_DONE.json ]; then echo "prefix already done $OUT"; exit 0; fi
else
  test -f $OUT/DONE.json && { echo "already DONE $OUT"; prefix_gc; exit 0; }
fi

source $D/sbatch/stage_ds.sh   # sets DP (train/dev only, never TEST)

J=${SLURM_JOB_ID:-nojob}; RC=${SLURM_RESTART_COUNT:-0}
if [ $KIND = prefix ]; then
  RES=""; ARGS="--stop_after_prefix"
  [ -f $OUT/prefix_state.pt ] && { echo "prefix_state.pt exists (atomic save) -> finishing"; ARGS=SKIP; }
  RUNPY=r25_run.py
else
  [ -f $PFX ] || [ -f $OUT/state_last.pt ] || { echo "R33_NO_PREFIX $PFX"; exit 3; }
  RES=$PFX; [ -f $OUT/state_last.pt ] && RES=$OUT/state_last.pt
  ARGS="--resume $RES"
  if [ $KIND = a1_iface ]; then RUNPY="r27_run.py --arm a1_iface --lam 0.0"; else RUNPY=r25_run.py; fi
  [ $MODE = smoke ] && ARGS="$ARGS --num_epochs ${R33_SMOKE_EPOCHS:-3}"
fi
N=1; [ -f $OUT/ATTEMPTS.tsv ] && N=$(( $(wc -l < $OUT/ATTEMPTS.tsv) + 1 ))
[ -f $OUT/RUN_$J.json ] && mv $OUT/RUN_$J.json $OUT/RUN_${J}_attempt$((N-1)).json   # a requeue keeps the job id
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$N" "$(date -u +%FT%TZ)" "$J" "$RC" "$(hostname)" "$GPU" "${RES:-from_scratch}" >> $OUT/ATTEMPTS.tsv
echo "attempt $N job $J restart_count $RC kind $KIND ${RES:-from_scratch}"
rc=0
if [ "$ARGS" != SKIP ]; then
  cd $CODE
  CMD="$PY -B -u $RUNPY --data_path $DP --data_name $DATA_NAME --noise_file $NOISE_FILE --noise_ratio $NOISE_RATIO --seed $SEED --output_dir $OUT $ARGS $SX"
  echo "CMD $CMD"
  if [ "$MODE" = formal ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    LIM=$(( $(squeue -j $SLURM_JOB_ID -h -o %L | awk -F'[-:]' '{ if (NF==4) print ($1*24+$2)*3600+$3*60+$4; else if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }') - 900 ))
    set +e; timeout $LIM $CMD; rc=$?; set -e
  else
    set +e; $CMD; rc=$?; set -e
  fi
fi
if [ $KIND = prefix ]; then
  if [ -f $OUT/prefix_state.pt ]; then
    rm -f $OUT/state_last.pt   # identical epoch-0 state; prefix_state.pt is the one children read
    printf '{"prefix_state": "%s", "bytes": %s, "sha256": "%s", "job": "%s", "utc": "%s"}\n' "$OUT/prefix_state.pt" \
      "$(stat -c %s $OUT/prefix_state.pt)" "$(sha256sum $OUT/prefix_state.pt | cut -d' ' -f1)" "$J" "$(date -u +%FT%TZ)" > $OUT/PREFIX_DONE.json.tmp
    mv $OUT/PREFIX_DONE.json.tmp $OUT/PREFIX_DONE.json
    echo "R33_PREFIX_DONE $CELL"; exit 0
  fi
elif [ -f $OUT/DONE.json ]; then echo "R33_UNIT_DONE $KIND $CELL"; prefix_gc; exit 0
elif [ "$MODE" = formal ]; then prefix_gc
fi
if [ "$MODE" = formal ] && [ $rc -eq 124 ] && [ $N -lt 8 ]; then
  NJ=$(sbatch --parsable --job-name=$SLURM_JOB_NAME --time=$(squeue -j $SLURM_JOB_ID -h -o %l) $D/sbatch/r33_unit.sbatch $KIND $CELL) || NJ=SUBMIT_FAILED_feeder_will_retry
  printf "%s\t%s\t%s\t%s\t%s\tcontinuation_of_%s\n" "$(date -u +%FT%TZ)" "$SLURM_JOB_NAME" "$NJ" "$KIND" "$CELL" "$SLURM_JOB_ID" >> $D/SUBMITTED.tsv
  echo "timeout: continuation $NJ"; exit 0
fi
[ "$MODE" = formal ] && printf "%s\t%s\trc=%s\tattempt=%s\n" "$(date -u +%FT%TZ)" "$J" "$rc" "$N" >> $OUT/FAILED.tsv
echo "R33_UNIT_FAILED $KIND $CELL rc=$rc attempt=$N"; exit 1
