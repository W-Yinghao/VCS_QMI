#!/usr/bin/env bash
#SBATCH --job-name=precheck_a2_feats
#SBATCH --partition=A100,H100,L40S,RTX6000PRO
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/precheck_a_wave2_feats_%x_%j.out
# Pre-check A wave 2: indoor -> outdoor identity splits + frozen CLIP features (GPU). OUT required.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${OUT:?}"
banner "pre-check A wave 2 features (indoor -> outdoor) -> ${OUT}"
export HF_HUB_OFFLINE=1
"$PYTHON" scripts/precheck_a_wave2.py features --out "$OUT" --workers 8 ${FEAT_ARGS:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
