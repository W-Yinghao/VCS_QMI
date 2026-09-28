#!/usr/bin/env bash
#SBATCH --job-name=watch_loop
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/watch_loop_%j.out
# Status watcher loop: regenerates reports/WATCH_status.md for STAGES every INTERVAL seconds (default 600).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${STAGES:?}"; INTERVAL=${INTERVAL:-600}
banner "watch loop: $STAGES every ${INTERVAL}s"
while :; do
  "$PYTHON" scripts/watch_status.py --output-root "$OUTPUT_ROOT" --out reports/WATCH_status.md --stages "$STAGES" 2>&1 | tail -2
  echo "[watch] $(date -u +%FT%TZ) refreshed"; sleep "$INTERVAL"
done
