#!/usr/bin/env bash
#SBATCH --job-name=precheck_a_ext_feats
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/precheck_a_ext_feats_%x_%j.out
# Pre-check A addendum 1: CLIP features for the Localized Narratives external target. DATA (LN dir) and OUT required; FEAT_ARGS optional (e.g. --limit 200).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${DATA:?}"; : "${OUT:?}"
banner "pre-check A external features <- ${DATA} -> ${OUT}"
export HF_HUB_OFFLINE=1
"$PYTHON" scripts/precheck_a_external.py features --data "$DATA" --out "$OUT" --workers ${SLURM_CPUS_PER_TASK:-4} ${FEAT_ARGS:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
