#!/usr/bin/env bash
#SBATCH --job-name=r6crema_gpu
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r6 GAIN-P1 CREMA-D on one GPU: args passed to scripts/r6_gain_crema.py (r6_av env: PyAV for in-process FLV decoding in the cost step).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
export PYTHON=/projects/EEG-foundation-model/yinghao/envs/r6_av/bin/python
echo "host=$(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) start=$(date -u +%FT%TZ)"
"$PYTHON" scripts/r6_gain_crema.py "$@"; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
