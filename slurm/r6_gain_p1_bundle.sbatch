#!/usr/bin/env bash
#SBATCH --job-name=r6gain_p1
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node52,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r6 GAIN-P1 full pilot in ONE GPU job: 2 families x 10 seeds (0-9) run sequentially (a system whose result.json exists is skipped, so a rerun
# resumes), then the aggregate.  All cost measurements come from the same GPU.  Arg 1: dataset (default avmnist).  Main env.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
DS=${1:-avmnist}; OUTD=/home/infres/yinwang/CS_QMI/outputs/R6_GAIN/$DS; rc=0
echo "host=$(hostname) $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ)"
for s in 0 1 2 3 4 5 6 7 8 9; do
  for f in brier ce; do
    if [ -f "$OUTD/${f}_s$s/result.json" ]; then echo "skip ${f}_s$s (result.json exists)"; continue; fi
    "$PYTHON" scripts/r6_gain_pilot.py run --dataset "$DS" --family $f --seed $s || rc=1
  done
done
"$PYTHON" scripts/r6_gain_pilot.py aggregate --dataset "$DS" || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
