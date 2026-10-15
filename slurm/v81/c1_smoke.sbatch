#!/usr/bin/env bash
#SBATCH --job-name=v81_c1_smoke
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --nice=0
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V8.1 C1 smoke on the real stack (scripts/v81/c1_train.py): CPU tests -> parameter inventory (committed before any unit) -> selftest (training
# modes, one-step mask check, V8 vs C1 s/step + peak memory on this GPU) -> 2-epoch x 40-step training with selection -> export (CAL_SELECT) -> FIT-AUDIT
# audit.  Outputs under outputs_v81/c1/smoke_<job> (never outputs_v8).  scripts/v81 snapshotted at job start.
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v81 || exit 2; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
SNAP=$(mktemp -d /tmp/v81_c1_${SLURM_JOB_ID}_XXXX); trap "rm -rf $SNAP" EXIT; mkdir -p $SNAP/scripts; cp -r scripts/v81 $SNAP/scripts/
PY="/projects/EEG-foundation-model/yinghao/envs/v7_fga/bin/python -I"; export TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
O=/projects/EEG-foundation-model/yinghao/outputs_v81/c1/smoke_${SLURM_JOB_ID}; mkdir -p $O; RES=/home/infres/yinwang/CS_QMI/ssl_pilot_v81/reports/V8_1_DIAGNOSTICS/results
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) code=$(cat $SNAP/scripts/v81/c1_train.py $SNAP/scripts/v81/v8_src/v8/*.py $SNAP/scripts/v81/v8_src/v7/*.py | sha256sum | cut -c1-16)"
step() { local name=$1; shift; local t=$(date +%s); "$@"; local r=$?; echo "=== $name rc=$r $(( $(date +%s) - t )) s $(date -u +%FT%TZ)"; return $r; }
step tests $PY -m pytest -q -p no:cacheprovider tests/v81/test_c1.py || { echo "exit=1 end=$(date -u +%FT%TZ)"; exit 1; }
step inventory $PY $SNAP/scripts/v81/c1_train.py inventory --out $RES/c1_param_inventory.json || { echo "exit=1 end=$(date -u +%FT%TZ)"; exit 1; }
step selftest $PY $SNAP/scripts/v81/c1_train.py selftest --out $RES/c1_selftest.json --arm U-CS --seed 0 --inventory $RES/c1_param_inventory.json || { echo "exit=1 end=$(date -u +%FT%TZ)"; exit 1; }
U=$O/V81-C1-U-CS-s0
step train $PY $SNAP/scripts/v81/c1_train.py train --arm U-CS --lr parent --seed 0 --unit-dir $U --max-steps 40 --eval-n 64 --epochs 2 || { echo "exit=1 end=$(date -u +%FT%TZ)"; exit 1; }
step export $PY $SNAP/scripts/v81/c1_train.py export --unit-dir $U --roles CAL_SELECT || { echo "exit=1 end=$(date -u +%FT%TZ)"; exit 1; }
step audit $PY $SNAP/scripts/v81/c1_train.py audit --unit-dir $U || { echo "exit=1 end=$(date -u +%FT%TZ)"; exit 1; }
ls -la $U $U/epochs $U/pred; echo "exit=0 end=$(date -u +%FT%TZ)"
