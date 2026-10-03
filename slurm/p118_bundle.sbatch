#!/usr/bin/env bash
#SBATCH --job-name=p118_bundle
#SBATCH --partition=A100,L40S,RTX6000PRO,H100
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P118 bundle (v5 NEXT-I-NEST): runs the units listed in BUNDLE ("CMD=truth|visual|i1audit ARGS='...'" per line) sequentially in one job.
# Finished outputs are skipped (truth: <out>.json; visual: per-run JSONs inside the runner; i1audit: <out>.json).  Normal QOS only.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${BUNDLE:?}"; banner "P118 bundle $BUNDLE"; rc_all=0
while IFS= read -r u || [ -n "$u" ]; do
  [[ -z "$u" || "$u" =~ ^# ]] && continue
  CMD=""; ARGS=""; eval "$u"; out=$(echo "$ARGS" | grep -oE -- '--out [^ ]+' | cut -d' ' -f2)
  if [ "$CMD" != visual ] && [ -n "$out" ] && [ -f "$out.json" ]; then echo "[skip] $CMD $out"; continue; fi
  case "$CMD" in truth) Sc=scripts/p118_truth.py ;; visual) Sc=scripts/p118_visual.py ;; i1audit) Sc=scripts/p109_i1_audit.py ;; *) echo "unknown $CMD"; rc_all=2; continue ;; esac
  echo "[unit] $(date -u +%FT%TZ) $CMD $ARGS"; "$PYTHON" $Sc $ARGS; rc=$?; echo "[unit-exit] rc=$rc $(date -u +%FT%TZ)"; [ $rc -ne 0 ] && rc_all=$rc
done < "$BUNDLE"
echo "exit=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
