#!/usr/bin/env bash
#SBATCH --job-name=layerprobe
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/layerprobe_%x_%j.out
# Task pre-check B-T1 (wave 2): closed-form VCS bilinear probe vs CKA, layer correspondence. REPORT required; SMOKE=1 for the probe; EXTRA_ARGS passed through.
# CPU is sufficient (feature extraction of 5k images + linear solves); pass -p <GPU partition> --gres=gpu:1 on the sbatch line for a GPU.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
: "${REPORT:?}"
banner "task pre-check B-T1 layer probe (smoke=${SMOKE:-0})"
if [ "${SMOKE:-0}" = "1" ]; then "$PYTHON" scripts/task_b_layer_probe.py --out "$REPORT" --smoke ${EXTRA_ARGS:-}; rc=$?; else "$PYTHON" scripts/task_b_layer_probe.py --out "$REPORT" ${EXTRA_ARGS:-}; rc=$?; fi
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
