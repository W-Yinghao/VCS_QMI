#!/bin/bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r27_audit_20261029/logs/%x_%j.out
# Usage: graphs.sbatch CELL NB OUTSUB   -- audit 1 + 2 on fixed FIT graphs of the cell's checkpoints (read-only replay).
# Checkpoints: early = common prefix (ID queue rebuilt in memory), mid = MISSING (not saved), final = a3_vcs state_last
# (+ a2_mean / a4_bce for f40_s1001 and f60_s1001). Only units with DONE.json. Rerunnable: finished labels are skipped.
set -euo pipefail
CELL=$1; NB=$2; SUB=$3
D=/home/infres/yinwang/CS_QMI/handoff/r27_audit_20261029
Q=/home/infres/yinwang/CS_QMI/handoff/r27_qtarget_20261029
RUNS=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_qtarget_20261029/runs/$CELL
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r27_audit_20261029/$SUB/$CELL
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
eval "$($PY - "$Q/CELLS.json" "$CELL" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))[sys.argv[2]]
for k in ('data_name', 'noise_ratio', 'seed', 'noise_file', 'prefix'):
    print(f"{k.upper()}={c[k]}")
PYEOF
)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $D/sbatch/stage_train.sh
mkdir -p $O
SPEC="early_prefix_reconstructed=$PREFIX mid=MISSING:mid-training_checkpoint_not_saved_by_R27"
ARMS="a3_vcs"; case $CELL in f40_s1001|f60_s1001) ARMS="a3_vcs a2_mean a4_bce";; esac
for a in $ARMS; do
  if [ -f $RUNS/$a/DONE.json ]; then SPEC="$SPEC final_$a=$RUNS/$a/state_last.pt"; else echo "not finished: $a (skipped; rerun later)"; fi
done
cd $D
$PY -B -u audit_graphs.py $O $CELL $NB $SPEC -- --data_path $DP --data_name $DATA_NAME --noise_ratio $NOISE_RATIO \
  --noise_file $NOISE_FILE --seed $SEED --output_dir $O/_opt_unused
echo GRAPHS_DONE $CELL
