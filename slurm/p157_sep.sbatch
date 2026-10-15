#!/usr/bin/env bash
#SBATCH --job-name=p157_sep
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P157 (SEP-2; reports/P157_SEP2_PROTOCOL_FROZEN_20261015.md).  Positional: run <seeds "200+201"> | pop.  Resumable (finished cells are skipped).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; cd /home/infres/yinwang/CS_QMI/ssl_pilot; O=/home/infres/yinwang/CS_QMI/outputs/P157_sep
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) code=$(cat src/vcs_estim/sep.py src/vcs_estim/p157.py scripts/p157_sep.py | sha256sum | cut -c1-16)"
case "$1" in
  run) "$PYTHON" scripts/p157_sep.py run --seeds "$2" --out $O; rc=$?;;
  pop) "$PYTHON" scripts/p157_sep.py pop --out $O; rc=$?;;
  *) echo "usage: run <seeds> | pop"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
