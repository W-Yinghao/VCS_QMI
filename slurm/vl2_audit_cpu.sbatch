#!/usr/bin/env bash
#SBATCH --job-name=vl2_audit
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL2 data audit + scene-input caches for RefCOCO and RefCOCO+ (main env).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; rc=0
for d in refcoco refcoco+; do echo "== $d"; VL_DATASET=$d "$PYTHON" scripts/vl2_00_audit.py || rc=1; done
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
