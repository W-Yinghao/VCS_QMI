#!/usr/bin/env bash
#SBATCH --job-name=vcs_p3_pilot
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/p3_pilot_%x_%j.out
# Usage: sbatch --export=ALL,METHOD=vcs p3_pilot.sbatch   (METHOD in vcs|simclr|vicreg); one GPU job at a time via dependencies.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${METHOD:?METHOD must be vcs|simclr|vicreg}"
RUN_ID="P3_${METHOD}_seed0"
banner "P3 pilot: ${METHOD} 20 epochs seed 0 -> ${RUN_ID}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
if [ -f "$RUN_DIR/checkpoints/last.pt" ] && [ "$("$PYTHON" -c "import json;print(json.load(open('$RUN_DIR/status.json'))['status'])" 2>/dev/null)" != "COMPLETED" ]; then
  echo "[p3] resuming $RUN_ID from last.pt (epoch boundary)"
  run_fg "$PYTHON" -m vcs_ssl.train --config configs/cifar10_pilot_${METHOD}.yaml --resume "$RUN_DIR/checkpoints/last.pt"; rc=$?
elif [ -f "$RUN_DIR/status.json" ] && [ "$("$PYTHON" -c "import json;print(json.load(open('$RUN_DIR/status.json'))['status'])" 2>/dev/null)" = "COMPLETED" ]; then
  echo "[p3] training already COMPLETED; skipping to evaluation"; rc=0
else
  run_fg "$PYTHON" -m vcs_ssl.train --config configs/cifar10_pilot_${METHOD}.yaml --run-id "$RUN_ID"; rc=$?
fi
if [ $rc -eq 0 ]; then
  run_fg "$PYTHON" -m vcs_ssl.evaluate --run-dir "$RUN_DIR" --checkpoint initial.pt --protocol pilot || rc=$?
  run_fg "$PYTHON" -m vcs_ssl.evaluate --run-dir "$RUN_DIR" --checkpoint epoch_020.pt --protocol pilot || rc=$?
fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
