#!/usr/bin/env bash
#SBATCH --job-name=p101_o_line
#SBATCH --partition=RTX6000PRO,H100,A100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p101_o_line_%j.out
# P101: O line on the six P84 frozen checkpoints (reference P35 VCS epoch 800 first), then the aggregate table.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
banner "P101 O line (frozen representations)"
"$PYTHON" scripts/o_line_frozen.py --out-dir /home/infres/yinwang/CS_QMI/outputs/P101_o_line --report reports/P101_o_line ${EXTRA:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
