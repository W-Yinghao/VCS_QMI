#!/bin/bash
#SBATCH --qos=runfill
#SBATCH --partition=V100-32GB,L40S,A100,A40
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101/logs/%x_%j.out
# NRCS-v1 GPU production gate (technical smoke, runfill allowed). Writes GATE_<job>.json and, if all pass, ADAPTER_READY.json.
set -euo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r32_native_ratio_20261101
O=/projects/EEG-foundation-model/yinghao/outputs_v9/r32_native_ratio_20261101
PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=$W/package/src:$W/nrcs:$W/upstream
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source $W/sbatch/stage_ims.sh
$PY -u -B $W/nrcs/gate.py --data $O/data --ims $IMS --out $W --smoke-dir $O/smoke
