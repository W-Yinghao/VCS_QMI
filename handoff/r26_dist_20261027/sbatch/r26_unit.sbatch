#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nice=30
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r26_dist_20261027/logs/%x_%j.out
# Usage: r26_unit.sbatch KIND CELL [ARM]
#   KIND prefix : native IN2R-code (pinned R25 r25_run.py) warmup + co-train epoch 0, saves prefix_state.pt
#   KIND native : native IN2R-code H0 full run from the cell prefix (stage C datasets without an R25 H0)
#   KIND arm    : R26 arm ARM full run from the cell prefix, c_J read from SCALE_<ds>.json (frozen)
#   KIND calib  : r26_calibrate.py on the cell prefix -> SCALE_<ds>.json
# Self-continuation: the python process runs under `timeout`; on timeout without DONE.json the job resubmits itself
# (same name/args, --resume state_last.pt), at most 3 attempts in total. No --dependency is used.
set -euo pipefail
KIND=$1; CELL=$2; ARM=${3:-}
D=/home/infres/yinwang/CS_QMI/handoff/r26_dist_20261027
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r26_dist_20261027
PY="/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -u"
PY0=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
eval "$($PY0 - "$D/CELLS.json" "$CELL" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))[sys.argv[2]]
for k in ('data_name', 'ds', 'noise_ratio', 'seed', 'noise_file', 'prefix'):
    print(f"{k.upper()}={c[k]}")
EOF
)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
SRC=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data/$DATA_NAME
case $DATA_NAME in cc152k_precomp) FILES="train_ims.npy dev_ims.npy train_caps.tsv dev_caps.tsv";; *) FILES="train_ims.npy dev_ims.npy train_caps.txt dev_caps.txt";; esac
NEED=$(cd $SRC && du -cb $FILES | tail -1 | cut -f1); AVAIL=$(df --output=avail -B1 "${TMPDIR:-/tmp}" | tail -1)
if [ "$AVAIL" -gt $(( NEED + 20*1024*1024*1024 )) ]; then
  LOC=$TMPDIR/r26_$SLURM_JOB_ID; mkdir -p $LOC/$DATA_NAME; trap 'rm -rf $TMPDIR/r26_$SLURM_JOB_ID' EXIT
  t0=$(date +%s); for f in $FILES; do cp $SRC/$f $LOC/$DATA_NAME/; [ $(stat -c %s $SRC/$f) = $(stat -c %s $LOC/$DATA_NAME/$f) ]; done
  DP=$LOC; echo "staged $NEED bytes in $(( $(date +%s)-t0 ))s"
else
  DP=/projects/EEG-foundation-model/yinghao/data/ncr_precomp/NCR-data/data; echo "no local space ($AVAIL < $NEED+20G): reading $SRC via mmap"
fi
cd $D/code
C="--data_path $DP --data_name $DATA_NAME --noise_file $NOISE_FILE --noise_ratio $NOISE_RATIO --seed $SEED"
case $KIND in
  prefix) OUT=$(dirname $PREFIX); mkdir -p $OUT
          $PY r25_run.py $C --output_dir $OUT --stop_after_prefix
          test -f $PREFIX; echo "R26_UNIT_DONE prefix $CELL"; exit 0;;
  calib)  $PY r26_calibrate.py $D/SCALE_$DS.json $C --resume $PREFIX --output_dir $O/calib/$CELL
          echo "R26_UNIT_DONE calib $CELL"; exit 0;;
  native) OUT=$O/runs/$CELL/H0; RUNPY="r25_run.py"; EXTRA="";;
  arm)    OUT=$O/runs/$CELL/$ARM; RUNPY="r26_run.py"
          CJ=$($PY0 -c "import json;print(json.load(open('$D/SCALE_$DS.json'))['c_J']['$ARM'])")
          [ "$CJ" != "None" ]; EXTRA="--arm $ARM --c_J $CJ";;
  *) echo "bad kind $KIND"; exit 2;;
esac
mkdir -p $OUT
test -f $OUT/DONE.json && { echo "already DONE $OUT"; exit 0; }
RES=$PREFIX; [ -f $OUT/state_last.pt ] && RES=$OUT/state_last.pt
N=$(cat $OUT/ATTEMPTS 2>/dev/null || echo 0); N=$((N+1)); echo $N > $OUT/ATTEMPTS
echo "attempt $N resume from $RES"
LIM=$(( $(squeue -j $SLURM_JOB_ID -h -o %L | awk -F'[-:]' '{ if (NF==4) print ($1*24+$2)*3600+$3*60+$4; else if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }') - 900 ))
set +e
timeout $LIM $PY $RUNPY $EXTRA $C --output_dir $OUT --resume $RES ${R26_TEST_EXTRA:-}
rc=$?
set -e
if [ -f $OUT/DONE.json ]; then echo "R26_UNIT_DONE $KIND $CELL $ARM"; exit 0; fi
if [ $rc -eq 124 ] && [ $N -lt 3 ]; then
  J=$(sbatch --parsable --job-name=$SLURM_JOB_NAME --time=$(squeue -j $SLURM_JOB_ID -h -o %l) $D/sbatch/r26_unit.sbatch $KIND $CELL $ARM)
  printf "%s\t%s\t%s\tcontinuation_of_%s\n" "$(date -u +%FT%TZ)" "$SLURM_JOB_NAME" "$J" "$SLURM_JOB_ID" >> $D/SUBMITTED.tsv
  echo "timeout: continuation submitted $J"; exit 0
fi
echo "R26_UNIT_FAILED rc=$rc attempt=$N"; exit 1
