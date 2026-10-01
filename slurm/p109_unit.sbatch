#!/usr/bin/env bash
#SBATCH --job-name=p109_unit
#SBATCH --partition=RTX6000PRO,H100,A100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P109 generic unit: CMD = one of x1 | i1feat | i1audit | i2, ARGS passed through (variables via the environment; no commas inside --export).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${CMD:?}"
case "$CMD" in
  x1) S=scripts/p109_x1.py ;; i1feat) S=scripts/p109_i1_features.py ;; i1audit) S=scripts/p109_i1_audit.py ;; i2) S=scripts/p109_i2_pilot.py ;;
  *) echo "unknown CMD $CMD"; exit 2 ;;
esac
banner "P109 $CMD: $ARGS"
"$PYTHON" $S $ARGS; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
