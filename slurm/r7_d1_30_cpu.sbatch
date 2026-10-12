#!/usr/bin/env bash
#SBATCH --job-name=r7_d1_30_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-30 D1-30 CPU units in the SparseVLM env (pyarrow / PIL for the sets; llava.eval.m4c_evaluator for TextVQA scoring).
# First arg: build (evaluation sets; refuses before the D1-30 freeze) | score | tests.  Positional arguments only (no --export).
E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm; D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1; R=$D1/d1_30
P=reports/R7/R7_D1_PROTOCOL_FROZEN_20261012.yaml; export PYTHONNOUSERSITE=1
cd /home/infres/yinwang/CS_QMI/ssl_pilot; what="$1"; shift
echo "host=$(hostname) start=$(date -u +%FT%TZ)"
case "$what" in
  build) $E/bin/python scripts/r7_d1_30_data.py build --out "$R/sets" --protocol "$P"; rc=$?;;
  score) $E/bin/python scripts/r7_d1_30_score.py --sets-dir "$R/sets" --pred-dir "$R/pred" --timing-dir "$R/timing" \
           --lr-choice "$D1/selectors/lr_choice.json" --out "$R/results" "$@"; rc=$?;;
  tests) $E/bin/python -m unittest tests/test_r7_d1_30.py -v; rc=$?;;
  *) echo "usage: build | score [--sparsevlm-primary budget|latency_b1] | tests"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
