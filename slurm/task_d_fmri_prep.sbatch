#!/usr/bin/env bash
#SBATCH --job-name=fmri_prep
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/task_d_fmri_prep_%x_%j.out
# P65 step 1: per-subject BOLD features at three motion-cleaning strengths. DATA (aomic func root), IDS, OUT required; PREP_ARGS passed through
# (e.g. "--smoke" or "--subjects sub-0002" — never put commas in --export values; pass variables via the environment).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
: "${DATA:?}"; : "${IDS:?}"; : "${OUT:?}"
banner "P65 fMRI prep: $DATA -> $OUT (${PREP_ARGS:-})"
"$PYTHON" scripts/task_d_fmri_prep.py --data "$DATA" --ids "$IDS" --out "$OUT" ${PREP_ARGS:-}; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
