#!/usr/bin/env bash
#SBATCH --job-name=v81_fork3_gpu
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --nice=400
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V8.1 fork 3 GPU diagnostics (scripts/v81/v81_fork3_gpu.py): timing -> infer -> interventions -> gradcheck, each resumable (done markers).
#   sbatch slurm/v81/v81_fork3_gpu.sbatch                 full run (STAGES=all)
#   STAGES="infer gradcheck" sbatch ...                   selected stages
#   SMOKE=1 sbatch --time=00:40:00 --job-name=v81_fork3_smoke slurm/v81/v81_fork3_gpu.sbatch     tiny-subset code-path check (separate output dir)
# Read-only on outputs_v8 (checkpoints loaded into an isolated model; nothing written back); outputs under outputs_v81/fork3[_smoke].
# scripts/v81 is snapshotted at job start so later edits never change a running job.
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v81 || exit 2; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
SNAP=$(mktemp -d /tmp/v81_fork3_${SLURM_JOB_ID}_XXXX); trap "rm -rf $SNAP" EXIT; mkdir -p $SNAP/scripts; cp -r scripts/v81 $SNAP/scripts/
PY="/projects/EEG-foundation-model/yinghao/envs/v7_fga/bin/python -I"
export TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
IDS=/home/infres/yinwang/CS_QMI/ssl_pilot_v81/reports/V8_1_DIAGNOSTICS/results/fit_audit_ids.json
STAGES=${STAGES:-all}; EXTRA=""; [ "${SMOKE:-0}" = "1" ] && { EXTRA="--smoke"; STAGES=${STAGES_SMOKE:-"infer interventions gradcheck"}; }
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) code=$(cat $SNAP/scripts/v81/v81_fork3_gpu.py $SNAP/scripts/v81/v8_src/v8/*.py $SNAP/scripts/v81/v8_src/v7/*.py | sha256sum | cut -c1-16) ids=$(sha256sum $IDS | cut -c1-16) stages='$STAGES' extra='$EXTRA'"
rc=0
for s in $STAGES; do
  $PY $SNAP/scripts/v81/v81_fork3_gpu.py $s --ids $IDS $EXTRA; rc=$?; echo "=== stage $s rc=$rc $(date -u +%FT%TZ)"
  [ $rc -ne 0 ] && break
done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
