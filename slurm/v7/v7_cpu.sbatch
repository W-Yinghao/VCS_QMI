#!/usr/bin/env bash
#SBATCH --job-name=v7_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V7 CPU steps (reports/V7/): tests | genai-prompts | audit.  Env: envs/v7_fga (python -I, no user site).  Positional arguments only.
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
E=/projects/EEG-foundation-model/yinghao/envs/v7_fga; PY="$E/bin/python -I"; O=/home/infres/yinwang/CS_QMI/outputs/V7; M=/projects/EEG-foundation-model/yinghao/models
echo "host=$(hostname) start=$(date -u +%FT%TZ) branch=$(git branch --show-current) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts/v7 tests/v7 slurm/v7 | wc -l)"
echo "code: $(cat scripts/v7/*.py tests/v7/*.py slurm/v7/*.sbatch | sha256sum | cut -c1-16) (sha256 of the V7 scripts, tests and wrappers)"
what="$1"; shift
case "$what" in
  tests) $PY -m pytest -q -p no:cacheprovider tests/v7 "$@"; rc=$?
         echo "--- V7 package reference tests (outputs/V7/package_ref/reference, read-only copy)"
         (cd /tmp && $PY -m pytest -q -p no:cacheprovider $O/package_ref/reference/tests); rc2=$?; [ $rc -eq 0 ] && rc=$rc2;;
  genai-prompts) $PY scripts/v7/v7_data.py genai-prompts --parquet-dir /projects/EEG-foundation-model/yinghao/data/genai_bench/data \
                   --out $O/data/external_prompts_genai_bench.jsonl; rc=$?;;
  audit) $PY scripts/v7/v7_data.py audit --out $O/data --external $O/data/external_prompts_genai_bench.jsonl --bert $M/bert-base-uncased "$@"; rc=$?;;
  n0-prep) BIG=/projects/EEG-foundation-model/yinghao/outputs_v7; $PY scripts/v7/v7_native.py prep --work $BIG/n0/work; rc=$?
           if [ $rc -eq 0 ]; then                                   # the smoke work dir: the first 70 natively mapped FIT records (code check only)
             mkdir -p $BIG/smoke/n0_work/dataset && $PY -c "import json,sys; d=json.load(open(sys.argv[1])); json.dump(d[:70], open(sys.argv[2],'w'))" \
               $BIG/n0/work/dataset/train_mask.json $BIG/smoke/n0_work/dataset/train_mask.json; rc=$?
           fi;;
  rank-inputs) $PY scripts/v7/v7_genai.py rank-inputs --out $O/data; rc=$?;;
  *) echo "usage: tests | genai-prompts | audit | n0-prep | rank-inputs"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
