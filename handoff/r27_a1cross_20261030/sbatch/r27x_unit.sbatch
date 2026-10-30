#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nice=30
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_a1cross_20261030/logs/%x_%j.out
# Usage: r27x_unit.sbatch CELL     R27 A1 (a1_iface, lam 0) full unit on COCO 40% / CC152K from the cell's R26 common
# prefix (CELLS_A1X.json: same prefix + noise map as R26), co-train epochs 1..39, code snapshot code_a1x.
# Self-continuation as R26/R27: python runs under `timeout`; on timeout without DONE.json the job resubmits itself
# (same name/args/time/nice, --resume state_last.pt), at most 3 attempts in total. No --dependency is used.
set -euo pipefail
CELL=$1; ARM=a1_iface; LAM=0.0
D=/home/infres/yinwang/CS_QMI/handoff/r27_a1cross_20261030
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_a1cross_20261030
PY="/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -B -u"
PY0=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
eval "$($PY0 -B - "$D/CELLS_A1X.json" "$CELL" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))[sys.argv[2]]
assert c['ds'] in ('coco', 'cc152k')
for k in ('data_name', 'noise_ratio', 'seed', 'noise_file', 'prefix'):
    print(f"{k.upper()}={c[k]}")
PYEOF
)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $D/sbatch/stage_ds.sh
cd $D/code_a1x
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
if [ -f $OUT/DONE.json ]; then echo "R27X_UNIT_DONE $CELL $ARM"; exit 0; fi
if [ $rc -eq 124 ] && [ $N -lt 3 ]; then
  J=$(sbatch --parsable --job-name=$SLURM_JOB_NAME --time=$(squeue -j $SLURM_JOB_ID -h -o %l) --nice=$(squeue -j $SLURM_JOB_ID -h -o %y) $D/sbatch/r27x_unit.sbatch $CELL)
  printf "%s\t%s\t%s\tcontinuation_of_%s\n" "$(date -u +%FT%TZ)" "$SLURM_JOB_NAME" "$J" "$SLURM_JOB_ID" >> $D/SUBMITTED.tsv
  echo "timeout: continuation submitted $J"; exit 0
fi
echo "R27X_UNIT_FAILED rc=$rc attempt=$N"; exit 1
