#!/usr/bin/env bash
#SBATCH --job-name=r7_d1full_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-30 D1-FULL CPU steps (SparseVLM env = the env of the frozen D1 manifest build; python -I for every read of downloaded data).
# First arg: tests | tests_main | check (tests + precheck) | build | verify | freeze | reuse [--copy] | plan | complete | data | precheck | choose | score [...].  Positional only (no --export).
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm; DF=/home/infres/yinwang/CS_QMI/outputs/R7_D1_FULL; D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
echo "host=$(hostname) start=$(date -u +%FT%TZ)"
what="$1"; shift
case "$what" in
  tests) "$E/bin/python" -m unittest tests/test_r7_d1full.py -v; rc=$?;;
  tests_main) "$PYTHON" -m unittest tests/test_r7_d1full.py -v; rc=$?;;
  check) "$E/bin/python" -m unittest tests/test_r7_d1full.py -v; rc=$?                 # tests, then the pre-freeze verification (dry-run files only)
         "$E/bin/python" -I scripts/r7_d1full_data.py precheck "$@"; rc2=$?; [ $rc -eq 0 ] && rc=$rc2;;
  build|verify|freeze|reuse|plan|complete|data|precheck) "$E/bin/python" -I scripts/r7_d1full_data.py "$what" "$@"; rc=$?;;
  choose) "$PYTHON" -I scripts/r7_d1full_plan.py choose --selectors-dir "$DF/selectors"; rc=$?;;
  score) "$E/bin/python" -I scripts/r7_d1full_score.py "$@"; rc=$?;;
  *) echo "usage: tests | tests_main | check | build | verify | freeze | reuse | plan | complete | data | precheck | choose | score"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
