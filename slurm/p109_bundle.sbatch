#!/usr/bin/env bash
#SBATCH --job-name=p109_bundle
#SBATCH --partition=RTX6000PRO,H100,A100,L40S
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P109 bundle (owner 2026-10-01: fewer SLURM slots; GPU where it accelerates): runs the units listed in BUNDLE (one "CMD=... ARGS='...'" per line) sequentially in one job.
# A unit whose --out file (<out>.json, or <out>/ for directories) already exists is skipped, so a re-submitted bundle resumes.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${BUNDLE:?}"
banner "P109 bundle $BUNDLE"
rc_all=0
while IFS= read -r u || [ -n "$u" ]; do
  [[ -z "$u" || "$u" =~ ^# ]] && continue
  CMD=""; ARGS=""; eval "$u"
  out=$(echo "$ARGS" | grep -oE -- '--out [^ ]+' | cut -d' ' -f2)
  if [ -n "$out" ] && { [ -f "$out.json" ] || { [ "$CMD" != i1audit ] && [ -f "$out" ]; }; }; then echo "[skip] $CMD $out exists"; continue; fi
  case "$CMD" in
    x1) Sc=scripts/p109_x1.py ;; i1feat) Sc=scripts/p109_i1_features.py ;; i1audit) Sc=scripts/p109_i1_audit.py ;; i2) Sc=scripts/p109_i2_pilot.py ;;
    *) echo "unknown CMD $CMD"; rc_all=2; continue ;;
  esac
  echo "[unit] $(date -u +%FT%TZ) $CMD $ARGS"
  "$PYTHON" $Sc $ARGS; rc=$?; echo "[unit-exit] rc=$rc $(date -u +%FT%TZ)"; [ $rc -ne 0 ] && rc_all=$rc
done < "$BUNDLE"
echo "exit=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
