#!/bin/bash
#SBATCH --job-name=te_eval
#SBATCH --qos=runfill
#SBATCH --partition=V100-32GB,V100,L40S,A100,A40,3090,A30
#SBATCH --exclude=node51,node53,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --nice=0
#SBATCH --requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/test_eval_20261101/logs/%x_%j.out
# TEST evaluation of frozen-manifest units: args HOST DATASET SHARD [extra test_eval.py args]. Re-runnable (done units skipped).
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
cd /home/infres/yinwang/CS_QMI/handoff/test_eval_20261101
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -B -u tools/test_eval.py --host $1 --dataset $2 --shard $3 "${@:4}"
echo TE_EVAL_DONE
