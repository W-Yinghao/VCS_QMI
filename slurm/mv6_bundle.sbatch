#!/usr/bin/env bash
#SBATCH --job-name=mv6_bundle
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=23:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# MV6 bundle (P142 / P143 / P144; CVPR-MV6-HANDOFF r2): runs the units listed in BUNDLE sequentially in one job, one per line:
#   CMD=feat|d45|i1audit|p142|p144|gate ARGS='...'  (gate = scripts/p143_gate.py; a failing gate fails the job)
# feat = scripts/p109_i1_features.py, d45 = scripts/precheck_d_features.py, i1audit = scripts/p109_i1_audit.py, p142 = scripts/p142_table3.py,
# p144 = scripts/p144_compress.py.  A unit whose output exists is skipped (i1audit / p142: <out>.json; feat: <out>/<run>/features.pt for every run;
# d45: <out>/manifest.json; p144: per-run JSONs inside the runner).  Normal QOS only.  Partition / GPU are set at submission.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
: "${BUNDLE:?}"; banner "MV6 bundle $BUNDLE"; rc_all=0
while IFS= read -r u || [ -n "$u" ]; do
  [[ -z "$u" || "$u" =~ ^# ]] && continue
  CMD=""; ARGS=""; eval "$u"; out=$(echo "$ARGS" | grep -oE -- '--out [^ ]+' | cut -d' ' -f2)
  case "$CMD" in
    feat) Sc=scripts/p109_i1_features.py; runs=$(echo "$ARGS" | grep -oE -- '--runs [^ ]+' | cut -d' ' -f2); done_all=1
          for r in ${runs//,/ }; do [ -f "$out/$r/features.pt" ] || done_all=0; done; [ $done_all = 1 ] && { echo "[skip] feat $runs"; continue; } ;;
    d45) Sc=scripts/precheck_d_features.py; [ -f "$out/manifest.json" ] && { echo "[skip] d45 $out"; continue; } ;;
    i1audit) Sc=scripts/p109_i1_audit.py; [ -f "$out.json" ] && { echo "[skip] i1audit $out"; continue; } ;;
    p142) Sc=scripts/p142_table3.py; [ -f "$out.json" ] && { echo "[skip] p142 $out"; continue; } ;;
    p144) Sc=scripts/p144_compress.py ;;
    gate) Sc=scripts/p143_gate.py ;;
    *) echo "unknown $CMD"; rc_all=2; continue ;;
  esac
  echo "[unit] $(date -u +%FT%TZ) $CMD $ARGS"; "$PYTHON" $Sc $ARGS; rc=$?; echo "[unit-exit] rc=$rc $(date -u +%FT%TZ)"; [ $rc -ne 0 ] && rc_all=$rc
done < "$BUNDLE"
echo "exit=$rc_all end=$(date -u +%FT%TZ)"; exit $rc_all
