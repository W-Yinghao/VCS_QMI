#!/usr/bin/env bash
#SBATCH --job-name=r7_d1_40_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-40 CPU steps.  First arg: tests (main env) | tests_vl (vl_baselines env: adds the transformers checks) | complete | readout [roles] |
# choose | imagecheck (manifest images present in the extracted GQA folder; no archive is ever read).  Positional arguments only (no --export).  DEV stays sealed (the readout refuses it without --allow-dev, R7-40c only).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
Q=/projects/EEG-foundation-model/yinghao/envs/vl_baselines; D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; R40=$D1/r7_40
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
FROZEN_SHA=32c72126ec9d40a362c67883f3a0b92a7212ae063b7ab9a369c47a696706d794
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts slurm tests | wc -l)"
echo "code: $(cat scripts/r7_d1_40_*.py tests/test_r7_d1_40.py slurm/r7_d1_40_* | sha256sum | cut -c1-16) (sha256 of the R7-40 scripts, tests and wrappers)"
what="$1"; shift
case "$what" in
  tests) "$PYTHON" -m unittest tests/test_r7_d1_40.py -v; rc=$?;;
  tests_vl) $Q/bin/python -m unittest tests/test_r7_d1_40.py -v; rc=$?;;
  complete) "$PYTHON" -I scripts/r7_d1_40_analyze.py complete --teacher-dir "$R40/teacher" --manifest-dir "$D1/data" --frozen-sha256 "$FROZEN_SHA"; rc=$?;;
  readout) "$PYTHON" -I scripts/r7_d1_40_analyze.py readout --teacher-dir "$R40/teacher" --manifest-dir "$D1/data" "$@"; rc=$?;;
  choose) "$PYTHON" -I scripts/r7_d1_40_plan.py choose --selectors-dir "$R40/selectors"; rc=$?;;
  imagecheck) "$PYTHON" -I scripts/r7_d1_40_teacher.py imagecheck --manifest-dir "$D1/data" --out "$R40/imagecheck" "$@"; rc=$?;;
  *) echo "usage: tests | tests_vl | complete | readout [--roles CAL FIT] | choose | imagecheck"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
