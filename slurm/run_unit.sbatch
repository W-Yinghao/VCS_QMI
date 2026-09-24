#!/usr/bin/env bash
#SBATCH --job-name=vcs_unit
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=23:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/unit_%x_%j.out
# Generic unit: sbatch --job-name=<name> --export=ALL,CFG=configs/x.yaml,RUN_ID=<id>,FINAL_CKPT=epoch_800.pt run_unit.sbatch
# Trains (resuming from last.pt at an epoch boundary if a previous attempt was interrupted), then evaluates initial.pt and FINAL_CKPT.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${CFG:?}"; : "${RUN_ID:?}"; : "${FINAL_CKPT:?}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
banner "unit ${RUN_ID}: cfg ${CFG} (sha $(sha256sum "$CFG" | cut -c1-12)) final ${FINAL_CKPT}"
status() { "$PYTHON" -c "import json;print(json.load(open('$RUN_DIR/status.json'))['status'])" 2>/dev/null; }
if [ -f "$RUN_DIR/checkpoints/last.pt" ] && [ "$(status)" != "COMPLETED" ]; then
  echo "[unit] resuming $RUN_ID from last.pt (epoch boundary; previous status $(status))"
  run_fg "$PYTHON" -m vcs_ssl.train --config "$CFG" --resume "$RUN_DIR/checkpoints/last.pt"; rc=$?
elif [ "$(status)" = "COMPLETED" ]; then
  echo "[unit] training already COMPLETED; skipping to evaluation"; rc=0
else
  run_fg "$PYTHON" -m vcs_ssl.train --config "$CFG" --run-id "$RUN_ID"; rc=$?
fi
if [ $rc -eq 0 ]; then
  [ -f "$RUN_DIR/evaluations/evaluation_initial.json" ] || run_fg "$PYTHON" -m vcs_ssl.evaluate --run-dir "$RUN_DIR" --checkpoint initial.pt --protocol pilot || rc=$?
  run_fg "$PYTHON" -m vcs_ssl.evaluate --run-dir "$RUN_DIR" --checkpoint "$FINAL_CKPT" --protocol pilot || rc=$?
fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
