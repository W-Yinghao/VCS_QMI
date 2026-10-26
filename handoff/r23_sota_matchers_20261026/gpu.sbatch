#!/usr/bin/env bash
#SBATCH --qos=normal
#SBATCH --partition=RTX6000PRO,L40S,H100
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nice=30
#SBATCH --no-requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r23_sota_matchers_20261026/logs/%x_%j.out
set -uo pipefail
export TZ=UTC PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home_sota
cd /home/infres/yinwang/CS_QMI/handoff/r23_sota_matchers_20261026
PY=/projects/EEG-foundation-model/yinghao/envs/sota_matchers/bin/python
MODE=$1; shift
if [ "$MODE" = smoke_all ]; then
  rc=0; for m in "$@"; do $PY -B harness.py smoke --method "$m" || { echo "SMOKE_ERROR $m"; rc=1; }; done; exit $rc
fi
exec $PY -B harness.py "$MODE" "$@"
