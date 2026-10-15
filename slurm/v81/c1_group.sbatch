#!/usr/bin/env bash
#SBATCH --job-name=v81_c1
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --nice=300
#SBATCH --signal=B:TERM@300
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V8.1 C1 formal units (V81_C1_PROTOCOL_FROZEN_20261015.yaml), grouped: sbatch slurm/v81/c1_group.sbatch U-CS:0 U-CS:1 U-CS:2
# Per unit: train (resumable; skipped when done.json exists) -> export CAL_SELECT / CAL_PROB / DEV (V8's export) -> FIT-AUDIT audit of best / resume.
# Outputs /projects/EEG-foundation-model/yinghao/outputs_v81/c1/units/V81-C1-<arm>-s<seed> (never outputs_v8).  scripts/v81 snapshotted at job start.
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v81 || exit 2; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
SNAP=$(mktemp -d /tmp/v81_c1g_${SLURM_JOB_ID}_XXXX); trap "rm -rf $SNAP" EXIT; mkdir -p $SNAP/scripts; cp -r scripts/v81 $SNAP/scripts/
PY="/projects/EEG-foundation-model/yinghao/envs/v7_fga/bin/python -I"; export TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
B=/projects/EEG-foundation-model/yinghao/outputs_v81/c1/units; mkdir -p $B
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) code=$(cat $SNAP/scripts/v81/c1_train.py $SNAP/scripts/v81/v8_src/v8/*.py $SNAP/scripts/v81/v8_src/v7/*.py | sha256sum | cut -c1-16) units='$*'"
rc=0
for spec in "$@"; do
  ARM=${spec%%:*}; SEED=${spec##*:}; U=$B/V81-C1-$ARM-s$SEED; t=$(date +%s); echo "[unit] $(date -u +%FT%TZ) $(basename $U)"
  $PY $SNAP/scripts/v81/c1_train.py train --arm $ARM --lr parent --seed $SEED --unit-dir $U; r=$?
  [ $r -eq 0 ] && [ -f $U/done.json ] || { echo "[unit-exit] $(basename $U) train rc=$r"; rc=1; continue; }
  if [ ! -f $U/pred/images_DEV.jsonl ]; then $PY $SNAP/scripts/v81/c1_train.py export --unit-dir $U --roles CAL_SELECT CAL_PROB DEV; r=$?; [ $r -ne 0 ] && { echo "[unit-exit] $(basename $U) export rc=$r"; rc=1; continue; }; fi
  if [ ! -f $U/pred/fit_audit.json ]; then $PY $SNAP/scripts/v81/c1_train.py audit --unit-dir $U; r=$?; [ $r -ne 0 ] && { echo "[unit-exit] $(basename $U) audit rc=$r"; rc=1; continue; }; fi
  echo "[unit-exit] $(basename $U) rc=0 $(( $(date +%s) - t )) s $(date -u +%FT%TZ)"
done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
