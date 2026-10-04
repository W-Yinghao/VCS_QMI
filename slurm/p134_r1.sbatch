#!/usr/bin/env bash
#SBATCH --job-name=p134_r1
#SBATCH --partition=RTX6000PRO,H100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p134_%x_%j.out
# P134 (R1 contamination): every cell of UNITS on one GPU (resumable). UNITS and OUT required; EXTRA optional (e.g. "--smoke --N 256 --updates 40").
#   sbatch --job-name=p134_s0 --export=ALL,UNITS=slurm/p134_units/seed0.txt,OUT=/home/infres/yinwang/CS_QMI/outputs/P134_r1 slurm/p134_r1.sbatch
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${UNITS:?}"; : "${OUT:?}"
banner "P134 R1 cells: $UNITS -> $OUT ${EXTRA:-}"
"$PYTHON" scripts/p134_r1.py --units "$UNITS" --out "$OUT" ${EXTRA:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
