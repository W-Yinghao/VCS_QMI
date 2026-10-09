#!/usr/bin/env bash
#SBATCH --job-name=vl1_12s_aggregate
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# VL1-12 addendum 2 (SigLIP 2) frozen readings; regression check that the generalised compare script reproduces the committed VL1-12 reading.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; rc=0; T=/home/infres/yinwang/CS_QMI/outputs/VL1_12_siglip2
echo "== regression (CLIP vs FG-CLIP 2, default args)"; "$PYTHON" scripts/vl1_12_compare.py --out $T/regression_vl1_12.json >/dev/null || rc=1
"$PYTHON" - <<'PY' || rc=1
import json
a = json.load(open("reports/VL1/VL1_12_results.json")); b = json.load(open("/home/infres/yinwang/CS_QMI/outputs/VL1_12_siglip2/regression_vl1_12.json")); b.pop("pair", None)
print("regression identical:", a == b); raise SystemExit(0 if a == b else 1)
PY
echo "== table"; "$PYTHON" scripts/vl1_10_aggregate.py --dir outputs/VL1_12_siglip2 --out reports/VL1/VL1_12s_table.json --ns all --routes vcs js softmax rff || rc=1
echo "== SigLIP 2 − CLIP"; "$PYTHON" scripts/vl1_12_compare.py --a clip:VL1_10 --b siglip2:VL1_12_siglip2 --out reports/VL1/VL1_12s_siglip2_minus_clip.json || rc=1
echo "== FG-CLIP 2 − SigLIP 2"; "$PYTHON" scripts/vl1_12_compare.py --a siglip2:VL1_12_siglip2 --b fgclip2:VL1_12 --out reports/VL1/VL1_12s_fgclip2_minus_siglip2.json || rc=1
echo "== strata"; "$PYTHON" scripts/vl1_10_strata.py --dir outputs/VL1_12_siglip2 --features /projects/EEG-foundation-model/yinghao/datasets/refcocog_umd/features_siglip2_base/train_side_features.pt --routes vcs js softmax --out reports/VL1/VL1_12s_strata.json || rc=1
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
