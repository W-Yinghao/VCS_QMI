#!/bin/bash
#SBATCH --job-name=te_layoutfree
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --nice=50
#SBATCH --requeue
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/test_eval_20261101/logs/%x_%j.out
# Layout-free TEST re-read (tools/test_eval_layoutfree.py). Args: "GPU_NAME" DATASET [extra args]. Partition / exclude given at
# submission so that the job runs on the GPU type recorded for the units (results/<sha16>.json env.gpu). Re-runnable (done units skipped).
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
cd /home/infres/yinwang/CS_QMI/handoff/test_eval_20261101
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -I -B -u tools/test_eval_layoutfree.py --gpu-name "$1" --dataset $2 "${@:3}" || exit $?
echo TE_LAYOUTFREE_DONE
