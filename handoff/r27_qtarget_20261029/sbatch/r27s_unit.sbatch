#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nice=30
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029/logs/%x_%j.out
# Usage: r27s_unit.sbatch CELL ARM   (R27-S arms s1..s6, code_v12; lam from LAMBDA_flickr.json, used by S1 only)
# Full R27 unit from the cell's common native prefix (CELLS.json, same prefixes as R25 H0 / R26), co-train epochs 1..39.
# lam: A1 = 0; A2-A5 = LAMBDA_flickr.json["lambda"] (frozen, one value for all four).
# Self-continuation as R26: python runs under `timeout`; on timeout without DONE.json the job resubmits itself
# (same name/args, --resume state_last.pt), at most 3 attempts in total. No --dependency is used.
set -euo pipefail
CELL=$1; ARM=$2
D=/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_qtarget_20261029
PY="/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -u"
PY0=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
eval "$($PY0 - "$D/CELLS.json" "$CELL" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))[sys.argv[2]]
assert c['ds'] == 'flickr'
for k in ('data_name', 'noise_ratio', 'seed', 'noise_file', 'prefix'):
    print(f"{k.upper()}={c[k]}")
PYEOF
)"
if [ "$ARM" = a1_iface ]; then LAM=0.0; else LAM=$($PY0 -c "import json;print(json.load(open('$D/LAMBDA_flickr.json'))['lambda'])"); fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $D/sbatch/stage.sh
cd $D/code_v12
OUT=$O/runs/$CELL/$ARM; mkdir -p $OUT
test -f $OUT/DONE.json && { echo "already DONE $OUT"; exit 0; }
RES=$PREFIX; [ -f $OUT/state_last.pt ] && RES=$OUT/state_last.pt
N=$(cat $OUT/ATTEMPTS 2>/dev/null || echo 0); N=$((N+1)); echo $N > $OUT/ATTEMPTS
echo "attempt $N resume from $RES lam $LAM"
LIM=$(( $(squeue -j $SLURM_JOB_ID -h -o %L | awk -F'[-:]' '{ if (NF==4) print ($1*24+$2)*3600+$3*60+$4; else if (NF==3) print $1*3600+$2*60+$3; else print $1*60+$2 }') - 900 ))
set +e
timeout $LIM $PY r27_run.py --arm $ARM --lam $LAM --data_path $DP --data_name $DATA_NAME --noise_file $NOISE_FILE \
  --noise_ratio $NOISE_RATIO --seed $SEED --output_dir $OUT --resume $RES
rc=$?
set -e
if [ -f $OUT/DONE.json ]; then echo "R27_UNIT_DONE $CELL $ARM"; exit 0; fi
if [ $rc -eq 124 ] && [ $N -lt 3 ]; then
  J=$(sbatch --parsable --job-name=$SLURM_JOB_NAME --time=$(squeue -j $SLURM_JOB_ID -h -o %l) --nice=$(squeue -j $SLURM_JOB_ID -h -o %y) $D/sbatch/r27s_unit.sbatch $CELL $ARM)
  printf "%s\t%s\t%s\tcontinuation_of_%s\n" "$(date -u +%FT%TZ)" "$SLURM_JOB_NAME" "$J" "$SLURM_JOB_ID" >> $D/SUBMITTED.tsv
  echo "timeout: continuation submitted $J"; exit 0
fi
echo "R27_UNIT_FAILED rc=$rc attempt=$N"; exit 1
