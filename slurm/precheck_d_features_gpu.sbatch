#!/usr/bin/env bash
#SBATCH --job-name=precheck_d_feat
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/precheck_d_feat_%x_%j.out
# Pre-check D feature extraction on one GPU. RUN, CKPT, OUT required; STRENGTHS (comma list, pass via the environment, not --export) and FEAT_ARGS optional.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${RUN:?}"; : "${CKPT:?}"; : "${OUT:?}"
banner "pre-check D features (GPU) ${RUN}/${CKPT} -> ${OUT}"
if [ -f "$OUT/manifest.json" ]; then echo "[precheck_d] features exist in $OUT; nothing to do"; exit 0; fi
"$PYTHON" scripts/precheck_d_features.py --run "$RUN" --checkpoint "$CKPT" --out "$OUT" --strengths "${STRENGTHS:-0,0.01,0.02,0.05,0.1,0.2,0.3,0.5}" --workers 6 ${FEAT_ARGS:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
