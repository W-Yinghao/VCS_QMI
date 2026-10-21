#!/usr/bin/env bash
# Phase A: unified EVAL_ONLY weight diagnostics with the package tool (v9/evidence/weight_diagnostics.py), one panel per seed.
# Main panel exactly as the package README: VCS / BCE / RuLSIF / RankMass (required) + ConstantU / ShuffleVCS (existing N2M controls).
# Production weights: the N2F shared stage (hashes frozen in N2F_FROZEN.json before N2M); controls: the N2M controls directory.
# `prepare` only hashes inputs (label bytes are hashed, never parsed); `evaluate` validates every weight file/lock before it parses labels.
# Outputs are new directories (the tool refuses to overwrite).  No training, no model selection, no production weight is written.
#   usage: run_weight_diagnostics.sh OUT_DIR
set -euo pipefail
OUT=${1:?output directory}
PY=/home/infres/yinwang/CS_QMI/env/bin/python
TOOLS=/home/infres/yinwang/CS_QMI/ssl_pilot_v9_n2p_ab/v9_n2p/VCS_V9_N2P_V10_Review_Server_Package_v1_20261020/v9/evidence
N2F=/projects/EEG-foundation-model/yinghao/outputs_v9/n2/f30k/rho0.4
N2M=/projects/EEG-foundation-model/yinghao/outputs_v9/n2m_20261019_v1/controls
mkdir -p "$OUT"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} start=$(date -u +%FT%TZ) tools_sha=$(sha256sum $TOOLS/weight_diagnostics.py | cut -c1-16)"
for s in 901 902 903; do
  S=$N2F/s$s
  T=$(sha256sum $S/teacher/teacher.pt | cut -d' ' -f1)          # actual shared teacher checkpoint (not exported)
  cd /tmp
  $PY -I $TOOLS/weight_diagnostics.py prepare \
    --observed $S/pairs.observed_pairs.json \
    --labels $S/eval_only/pairs.eval_only.json \
    --noise-manifest $S/pairs.noise_manifest.json \
    --teacher-sha256 "$T" --seed $s \
    --method "VCS=$S/weights/VCS_Weight.csv" \
    --method "BCE=$S/weights/BCE_Weight.csv" \
    --method "RuLSIF=$S/weights/RuLSIF_Weight.csv" \
    --method "RankMass=$S/weights/RankMass.csv" \
    --method "ConstantU=$N2M/s$s/ConstantU.csv" \
    --method "ShuffleVCS=$N2M/s$s/ShuffleVCS.csv" \
    --out "$OUT/diagnostic_input_s$s.json"
  $PY -I $TOOLS/weight_diagnostics.py evaluate --config "$OUT/diagnostic_input_s$s.json" --out "$OUT/diagnostics_s$s"
done
echo "end=$(date -u +%FT%TZ)"
