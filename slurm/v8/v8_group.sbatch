#!/usr/bin/env bash
#SBATCH --job-name=v8_group
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V8 group job (V8-03 FROZEN): units one after the other on one GPU; each arg "ROLE:ARM:LR:SEED" (ROLE TUNE | FIT).  The code (scripts/v8 and the
# V7 modules it imports) is snapshotted at job start, so later edits never change a running group.  Units are resumable and skip when done.
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v8; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
SNAP=$(mktemp -d /tmp/v8_group_${SLURM_JOB_ID}_XXXX); trap "rm -rf $SNAP" EXIT; mkdir -p $SNAP/scripts; cp -r scripts/v8 scripts/v7 $SNAP/scripts/
PY="/projects/EEG-foundation-model/yinghao/envs/v7_fga/bin/python -I"; O=/projects/EEG-foundation-model/yinghao/outputs_v8/units; mkdir -p $O
export TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) code=$(cat $SNAP/scripts/v8/*.py | sha256sum | cut -c1-16)"
rc=0
for spec in "$@"; do IFS=: read -r role arm lr seed <<< "$spec"
  if [ "$role" = TUNE ]; then u=$O/V8-TUNE-$arm-lr$lr; else u=$O/V8-FIT-$arm-s$seed; fi
  echo "[group] $role $arm lr $lr seed $seed -> $(basename $u) start $(date -u +%FT%TZ)"
  $PY $SNAP/scripts/v8/v8_train.py train --arm $arm --lr $lr --seed $seed --unit-dir $u --workers 8 && $PY $SNAP/scripts/v8/v8_train.py export --unit-dir $u \
    --rank-inputs /home/infres/yinwang/CS_QMI/outputs/V7/data/genai_rank_inputs.jsonl; r=$?
  echo "[group] $(basename $u) done (rc $r) $(date -u +%FT%TZ)"; [ $r -eq 0 ] || rc=1
done; echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
