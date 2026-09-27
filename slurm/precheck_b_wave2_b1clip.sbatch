#!/usr/bin/env bash
#SBATCH --job-name=b1clip
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/b1clip_%x_%j.out
# Pre-check B-S1 (wave 2): closed form vs trained cosine critic on CLIP adapter features. REPORT required; FEATURES defaults to P49; SMOKE=1 for the probe.
# CPU is sufficient (adapter training + linear solves); pass -p <GPU partition> --gres=gpu:1 on the sbatch line to run on a GPU.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
: "${REPORT:?}"
banner "pre-check B-S1 closed form on CLIP features (smoke=${SMOKE:-0})"
if [ "${SMOKE:-0}" = "1" ]; then "$PYTHON" scripts/precheck_b1_clip.py --features "${FEATURES:-$OUTPUT_ROOT/P49_precheck_A}" --out "$REPORT" --smoke; rc=$?; else "$PYTHON" scripts/precheck_b1_clip.py --features "${FEATURES:-$OUTPUT_ROOT/P49_precheck_A}" --out "$REPORT" ${EXTRA_ARGS:-}; rc=$?; fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
