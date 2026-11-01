#!/bin/bash
#SBATCH --job-name=te_manifest
#SBATCH --qos=normal
#SBATCH --partition=CPU,cpu-high
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/test_eval_20261101/logs/%x_%j.out
# Build + freeze the initial TEST manifest (TEST files are only stat()ed). Hashes every unit checkpoint.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /home/infres/yinwang/CS_QMI/handoff/test_eval_20261101
/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python -B -u tools/build_manifest.py --initial
