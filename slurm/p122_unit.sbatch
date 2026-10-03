#!/usr/bin/env bash
#SBATCH --job-name=p122_unit
#SBATCH --partition=A100,L40S,RTX6000PRO,H100
#SBATCH --exclude=node51
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# P122 (v6 V6-EVIDENCE) bundle job: one dataset, the encoders in RUNS, the measurement laws in LAWS, sequentially in ONE GPU job; finished
# (encoder, law) JSONs are skipped, augmented images and features are cached (outputs/P122_cache).  Env: DATASET, RUNS (comma list), LAWS
# (comma list), OUT (default outputs/P122_evidence), EXTRA (e.g. --smoke).  Normal QOS only (never runfill); node60 allowed.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
: "${DATASET:?}" "${RUNS:?}"
OUT=${OUT:-/home/infres/yinwang/CS_QMI/outputs/P122_evidence}
banner "P122 evidence: $DATASET laws=${LAWS:-standard} runs=$RUNS -> $OUT ${EXTRA:-}"
"$PYTHON" scripts/p122_evidence.py --dataset "$DATASET" --runs "$RUNS" --laws "${LAWS:-standard}" --out "$OUT" ${EXTRA:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
