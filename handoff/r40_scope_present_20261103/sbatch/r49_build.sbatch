#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --output=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103/logs/%x_%j.out
# R49 prep: R47 layout-free parent cache / prep / JOINT preps (tools/r49_build_lf.py = r47_build_lf.py with the Flickr-tree test extended to r49f) for each listed base parent, in order ($1 = tree, $2.. = bases).
set -uo pipefail
W=/home/infres/yinwang/CS_QMI/handoff/r40_scope_present_20261103; PY=/projects/EEG-foundation-model/yinghao/envs/v9_sgr/bin/python
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
nvidia-smi --query-gpu=name --format=csv; T=$1; shift; fail=0
for b in "$@"; do echo "=== $(date -u +%FT%TZ) $T $b"; $PY -u -B $W/tools/r49_build_lf.py $T $b || { echo "R49_BUILD_FAILED $b"; fail=1; }; done
echo "R49_BUILD_END fail=$fail $(date -u +%FT%TZ)"; exit $fail
