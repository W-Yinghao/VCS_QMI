#!/usr/bin/env bash
#SBATCH --job-name=r7_fix_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r7 R7-20 CPU steps.  First arg: tests | tests_vl | d3 | d2text | d1build | tables; the rest passed through.
# Main env for tests / d3 / d2text; the SparseVLM env (pyarrow, official llava code) for tests_vl / d1build.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm; OUT=/home/infres/yinwang/CS_QMI/outputs/R7_fixtures; FX=/projects/EEG-foundation-model/yinghao/data/r7_fixtures
mkdir -p "$OUT" "$FX"; [ -e "$OUT/fixture" ] || ln -s "$FX" "$OUT/fixture"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1   # keep ~/.local out of both envs
what="$1"; shift
case "$what" in
  tests) "$PYTHON" -m unittest tests/test_r7_fixtures.py -v; rc=$?;;
  tests_vl) "$E/bin/python" -m unittest tests/test_r7_fixtures.py -v; rc=$?;;
  d3) "$PYTHON" scripts/r7_fixtures_d3.py --out /home/infres/yinwang/CS_QMI/outputs/R7_D3 --workers 8 "$@"; rc=$?;;
  d2text) "$PYTHON" scripts/r7_fixtures_d2.py text --out "$OUT" "$@"; rc=$?;;
  d1build) "$E/bin/python" scripts/r7_fixtures_d1.py build --out "$OUT" "$@"; rc=$?;;
  tables) "$PYTHON" scripts/r7_fixtures_tables.py; rc=$?;;
  *) echo "usage: tests | tests_vl | d3 | d2text | d1build | tables"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
