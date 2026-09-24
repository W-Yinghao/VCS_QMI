#!/usr/bin/env bash
#SBATCH --job-name=vcs_p5_confirm
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p5_confirm_%j.out
# Usage: sbatch --export=ALL,METHOD=vcs,SEED=0 p5_confirm.sbatch  -> run P5_<method>_seed<seed> (200 epochs) + frozen evals (initial, epoch_200)
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${METHOD:?METHOD must be vcs|simclr|vicreg}"; : "${SEED:?SEED must be 0|1|2}"
CFG="configs/cifar10_confirm200_${METHOD}_seed${SEED}.yaml"
RUN_ID="P5_${METHOD}_seed${SEED}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
banner "P5 confirm200: ${METHOD} seed ${SEED} -> ${RUN_ID} (config sha $(sha256sum "$CFG" | cut -c1-12))"
status() { "$PYTHON" -c "import json,sys;print(json.load(open('$RUN_DIR/status.json'))['status'])" 2>/dev/null; }
if [ -f "$RUN_DIR/checkpoints/last.pt" ] && [ "$(status)" != "COMPLETED" ]; then
  echo "[p5] resuming $RUN_ID from last.pt (epoch boundary; previous status $(status))"
  run_fg "$PYTHON" -m vcs_ssl.train --config "$CFG" --resume "$RUN_DIR/checkpoints/last.pt"; rc=$?
elif [ "$(status)" = "COMPLETED" ]; then
  echo "[p5] training already COMPLETED; skipping to evaluation"; rc=0
else
  run_fg "$PYTHON" -m vcs_ssl.train --config "$CFG" --run-id "$RUN_ID"; rc=$?
fi
if [ $rc -eq 0 ]; then
  [ -f "$RUN_DIR/evaluations/evaluation_initial.json" ] || run_fg "$PYTHON" -m vcs_ssl.evaluate --run-dir "$RUN_DIR" --checkpoint initial.pt --protocol pilot || rc=$?
  run_fg "$PYTHON" -m vcs_ssl.evaluate --run-dir "$RUN_DIR" --checkpoint epoch_200.pt --protocol pilot || rc=$?
fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
