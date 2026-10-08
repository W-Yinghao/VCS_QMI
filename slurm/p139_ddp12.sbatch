#!/usr/bin/env bash
#SBATCH --job-name=p139_ddp12
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p139_ddp_%j.out
# P139 end-to-end DDP throughput on ONE node: 1 and 2 GPUs (4-GPU = p139_ddp.sbatch, deferred) in sequence (CUDA_VISIBLE_DEVICES subsets), ResNet-50 224 px, 2 views,
# 128 images per GPU, SyncBN, A-P3 (vcs) and SimCLR; 10 warm-up + 30 measured steps per cell; no checkpoints.  Also checks global image-UID
# uniqueness every step and times the gradient all-reduce / token all-gather.  Partition override: sbatch --partition=H100 slurm/p139_ddp.sbatch
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
set -o pipefail
banner "P139 DDP throughput 1/2 GPUs"
TORCHRUN=/home/infres/yinwang/CS_QMI/env/bin/torchrun; rc=0
for N in 1 2; do
  DEV=$(seq -s, 0 $((N-1)))
  CUDA_VISIBLE_DEVICES=$DEV $TORCHRUN --standalone --nproc_per_node=$N scripts/p139_imagenet_pipeline.py ddp --workers $(( SLURM_CPUS_PER_TASK / N - 2 < 16 ? SLURM_CPUS_PER_TASK / N - 2 : 16 )) \
    --out reports/P139/ddp_${SLURM_JOB_PARTITION:-x}_${N}gpu_${SLURM_JOB_ID}.json 2>&1 | grep -vE "UserWarning|self.setter|warnings.warn" | tail -8 || rc=1
done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
