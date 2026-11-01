#!/bin/bash
#SBATCH --qos=runfill
#SBATCH --partition=A100,A40,L40S,V100-32GB
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r33_baselines_20261102/logs/%x_%j.out
# Usage: r33_smoke.sbatch CELL   prefix (max_iters 20) -> each arm 2 co-train epochs from the prefix -> each arm resumes
# from its own state_last.pt for 1 more epoch -> tools/check_smoke.py. Outputs under outputs/.../smoke/<cell>/.
set -euo pipefail
CELL=$1; D=/home/infres/yinwang/CS_QMI/handoff/r33_baselines_20261102; S=/projects/EEG-foundation-model/yinghao/outputs_v9/r33_baselines_20261102/smoke/$CELL
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
rm -rf $S
U=$D/sbatch/r33_unit.sh
bash $U prefix $CELL smoke
ARMS=$(/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -B -c "import json;print(' '.join(json.load(open('$D/CELLS.json'))['$CELL']['arms']))")
for A in $ARMS; do
  R33_SMOKE_EPOCHS=3 bash $U $A $CELL smoke
  mv $S/$A/DONE.json $S/$A/DONE_first.json; cp $S/$A/EPOCHS.json $S/$A/EPOCHS_first.json
  R33_SMOKE_EPOCHS=4 bash $U $A $CELL smoke
done
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -B $D/tools/check_smoke.py $CELL $S "$ARMS" | tee $D/SMOKE_${CELL}_${SLURM_JOB_ID}.json
