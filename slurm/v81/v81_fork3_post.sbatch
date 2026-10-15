#!/usr/bin/env bash
#SBATCH --job-name=v81_fork3_post
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V8.1 fork 3 post-processing (CPU, a few minutes): GPU summaries (+ param_audit.json when present) -> results/{checkpoint_inference,input_checks,
# gradient_checks}.{json,csv} and V81_05_INPUT_GRADIENT_CHECKS.md; every inference row file re-validated with the package validator.
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v81 || exit 2; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) param_audit=$([ -f reports/V8_1_DIAGNOSTICS/results/param_audit.json ] && echo present || echo absent)"
/projects/EEG-foundation-model/yinghao/envs/v7_fga/bin/python -I scripts/v81/v81_fork3_post.py; rc=$?
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
