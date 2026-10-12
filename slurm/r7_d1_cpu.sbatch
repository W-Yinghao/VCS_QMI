#!/usr/bin/env bash
#SBATCH --job-name=r7_d1_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# R7-30 D1 CPU steps.  First arg: tests | tests_vl | manifest | shards | feed | zipcheck | analyze; the rest passed through.
# Main env for tests / analyze; the SparseVLM env (pyarrow, official llava code) for tests_vl and for manifest / zipcheck when a parquet source is used.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
E=/projects/EEG-foundation-model/yinghao/envs/sparsevlm; D1=/home/infres/yinwang/CS_QMI/outputs/R7_D1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
what="$1"; shift
case "$what" in
  tests) "$PYTHON" -m unittest tests/test_r7_d1.py tests/test_r7_fixtures.py -v; rc=$?;;
  tests_vl) "$E/bin/python" -m unittest tests/test_r7_d1.py tests/test_r7_fixtures.py -v; rc=$?;;
  manifest) "$E/bin/python" scripts/r7_d1_data.py manifest --out "$D1/data" "$@"; rc=$?;;
  shards) "$PYTHON" scripts/r7_d1_data.py shards --out "$D1/data" "$@"; rc=$?;;
  feed) "$PYTHON" scripts/r7_d1_data.py feed --out "$D1/data" --lines slurm/r7_d1_10_feed_lines.txt; rc=$?;;
  zipcheck) "$PYTHON" scripts/r7_d1_data.py zipcheck --out "$D1/data" --zip /projects/EEG-foundation-model/yinghao/data/gqa/zips/images.zip; rc=$?;;
  analyze) "$PYTHON" scripts/r7_d1_analyze.py --teacher-dir "$D1/teacher" --manifest-dir "$D1/data" "$@"; rc=$?;;
  *) echo "usage: tests | tests_vl | manifest | shards | feed | zipcheck | analyze"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
