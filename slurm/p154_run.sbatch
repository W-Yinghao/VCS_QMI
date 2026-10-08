#!/usr/bin/env bash
#SBATCH --job-name=p154_run
#SBATCH --partition=RTX6000PRO,H100
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=23:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P154 (FROZEN): ImageNet-100 pretraining (resumes from last.pt), then the readout of the epoch-50 / 100 / 200 checkpoints.
#   sbatch --job-name=p154_<m> slurm/p154_run.sbatch <vcs|js|simclr> <run_id>
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
M=$1; R=$2; rc=0
nvidia-smi --query-gpu=name --format=csv,noheader; echo "host $(hostname) start $(date -u +%FT%TZ)"
"$PYTHON" scripts/p154_in100.py train --method $M --run-id $R --epochs 200 --workers 30 || rc=$?
if [ $rc -eq 0 ]; then for e in 50 100 200; do [ -f $OUTPUT_ROOT/$R/evaluations/evaluation_epoch_$e.json ] || "$PYTHON" scripts/p154_in100.py eval --run-id $R --checkpoint epoch_$e.pt --workers 30 || rc=$?; done; fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
