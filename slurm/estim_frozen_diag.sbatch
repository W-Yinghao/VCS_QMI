#!/usr/bin/env bash
#SBATCH --job-name=estim_pkg_frozen_diag
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/estim_pkg_frozen_diag_%j.out
# Estimator package v1 §8: frozen-checkpoint estimator diagnostics, one JSON per checkpoint + aggregate table.
# OUT_DIR (default outputs/estim_pkg_frozen_diag) and REPORT (default reports/ESTIM_frozen_diag.md) via the environment.
# SMOKE=1: CPU partition probe (submit with -p CPU --gres=gpu:0): one checkpoint, tiny roles, 100 updates, gradient diagnostic on.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
O=/home/infres/yinwang/CS_QMI/outputs; OUT_DIR=${OUT_DIR:-$O/estim_pkg_frozen_diag}; REPORT=${REPORT:-reports/ESTIM_frozen_diag.md}
mkdir -p "$OUT_DIR"
banner "estimator package §8 frozen diagnostics (smoke=${SMOKE:-0}) -> $OUT_DIR"
if [ "${SMOKE:-0}" = "1" ]; then
  UNITS="P35_vcs_a5_views4_800ep_seed0:epoch_800"; EXTRA="--smoke --cpu --grad-diag --workers 4"
else
  UNITS="P35_vcs_a5_views4_800ep_seed0:initial P35_vcs_a5_views4_800ep_seed0:epoch_100 P35_vcs_a5_views4_800ep_seed0:epoch_400 P35_vcs_a5_views4_800ep_seed0:epoch_800
P41_simclr_views4_800ep_seed0:initial P41_simclr_views4_800ep_seed0:epoch_100 P41_simclr_views4_800ep_seed0:epoch_400 P41_simclr_views4_800ep_seed0:epoch_800
P5_vcs_seed0:initial P5_vcs_seed0:epoch_200"; EXTRA="--workers 7"
fi
rc=0; outs=()
for u in $UNITS; do
  run=${u%%:*}; ck=${u##*:}; out="$OUT_DIR/${run}__${ck}.json"
  if [ -s "$out" ]; then echo "skip (exists): $out"; outs+=("$out"); continue; fi
  "$PYTHON" -m vcs_estim.frozen --stage frozen_diagnostics --run-dir "$O/$run" --ckpt "$ck" --out "$out" $EXTRA || { rc=1; echo "FAILED $u"; continue; }
  outs+=("$out")
done
[ ${#outs[@]} -gt 0 ] && "$PYTHON" -m vcs_estim.frozen --stage aggregate --inputs "${outs[@]}" --out "$REPORT"
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
