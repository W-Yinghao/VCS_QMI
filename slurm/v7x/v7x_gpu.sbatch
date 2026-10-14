#!/usr/bin/env bash
#SBATCH --job-name=v7x_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V7X GPU jobs (reports/V7/V7X_*): env envs/v7_fga (torch 2.9 cu128 -> RTX6000PRO / H100 / L40S).  Positional arguments only.
#   x20-smoke                 X20 extractor on 64 images of each role (code check + invariant + timing; no result is used)
#   x20                       X20 common features: FIT / CAL_SELECT / CAL_PROB / DEV
#   x20-rank                  X20 common features of the GenAI-Rank candidates (parser v3 elements)
#   x21                       X21: all common-feature fits (tuning -> lr -> formal seeds, kernels, A0) on $BIG/x20
#   x21-pca <K>               X21 shared sensitivity: the same fits on FIT-only PCA-K features (every method alike) -> $BIG/x21_pca<K>
#   tifa-qg [limit]           B4T question generation + filter for the 800 GenAI-Rank prompts (env v7_tifa)
#   tifa-score <k/n> [limit]  B4T TIFA scores, shard k of n (env v7_tifa; images staged node-local)
#   dsg-gen <k/n> [limit]     B4D DSG generation with Qwen2.5-32B-Instruct, shard k of n (env v7x_llm; >= 80 GB GPU)
#   dsg-score <k/n> [limit]   B4D DSG scores with mPLUG-large, shard k of n (env v7_tifa)
#   b2-rank <k/n> [limit]     B2 VQAScore on the Qwen2.5-VL-7B host (t2v-metrics code, B1 templates), shard k of n (env v7x_llm)
#   hpsv3-rank <k/n> [limit] B5 HPSv3 on the GenAI-Rank candidates (env v7x_hps)
#   dev-<m> ...               the same scorers on EvalMuse DEV (secondary table; inputs from x30_dev_inputs.py, images staged, outputs $BIG/x30_dev):
#                             dev-tifa-qg | dev-tifa-score k/n | dev-dsg-gen k/n | dev-dsg-score k/n | dev-pnvqa-gen k/n own|evalmuse |
#                             dev-pnvqa-score k/n own|evalmuse | dev-b1 | dev-b2 | dev-hpsv3
#   pnvqa-gen <k/n> <own|parser> [limit]   B3 / B3-P element + question generation with Qwen2.5-32B-Instruct (env v7x_llm; >= 80 GB GPU)
#   pnvqa-score <k/n> <own|parser> [limit] B3 / B3-P scores with Qwen2-VL-7B-Instruct (env v7x_llm)
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
E=/projects/EEG-foundation-model/yinghao/envs/v7_fga; PY="$E/bin/python -I"; BIG=/projects/EEG-foundation-model/yinghao/outputs_v7
export TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) branch=$(git branch --show-current) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts/v7x tests/v7x slurm/v7x | wc -l)"
echo "code: $(cat scripts/v7x/*.py tests/v7x/*.py slurm/v7x/*.sbatch 2>/dev/null | sha256sum | cut -c1-16) (V7X scripts, tests, wrappers); v1 code: $(cat scripts/v7/*.py tests/v7/*.py slurm/v7/*.sbatch | sha256sum | cut -c1-16)"
what="$1"; shift
case "$what" in
  x20-smoke) $PY scripts/v7x/x20_features.py extract --out $BIG/x20_smoke --roles FIT CAL_SELECT CAL_PROB DEV --limit 64; rc=$?;;
  x20) $PY scripts/v7x/x20_features.py extract --out $BIG/x20; rc=$?;;
  x20-rank) $PY scripts/v7x/x20_features.py extract-rank --out $BIG/x20; rc=$?;;
  x21) $PY scripts/v7x/x21_fit.py all --x20 $BIG/x20 --out $BIG/x21; rc=$?;;
  x21-pca) $PY scripts/v7x/x21_fit.py all --x20 $BIG/x20 --out $BIG/x21_pca$1 --pca $1; rc=$?;;
  tifa-qg|tifa-score|dsg-score)
     T=/projects/EEG-foundation-model/yinghao/envs/v7_tifa/bin/python; X=$BIG/x30; mkdir -p $X; export HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf_cache_v7x MODELSCOPE_CACHE=$X/modelscope_cache
     export V7X_LOCAL=/tmp/v7x_${SLURM_JOB_ID}; mkdir -p $V7X_LOCAL; trap "rm -rf $V7X_LOCAL" EXIT; D=/home/infres/yinwang/CS_QMI/outputs/V7/data
     case "$what" in
       tifa-qg) $T -I scripts/v7x/x30_tifa.py qg --prompts $D/genai_rank_prompts.jsonl --out $X/tifa_qa.jsonl --limit ${1:-0}; rc=$?;;
       tifa-score) $T -I scripts/v7x/x30_tifa.py score --qa $X/tifa_qa.jsonl --inputs $D/genai_rank_inputs.jsonl --out $X/tifa_scores_${1/\//of}.jsonl --shard $1 --limit ${2:-0}; rc=$?;;
       dsg-score) $T -I scripts/v7x/x30_dsg.py score --dsg $X/dsg_gen_*.jsonl --inputs $D/genai_rank_inputs.jsonl --out $X/dsg_scores_${1/\//of}.jsonl --shard $1 --limit ${2:-0}; rc=$?;;
     esac;;
  b2-rank) L=/projects/EEG-foundation-model/yinghao/envs/v7x_llm/bin/python; X=$BIG/x30; mkdir -p $X; export V7X_LOCAL=/tmp/v7x_${SLURM_JOB_ID}; mkdir -p $V7X_LOCAL; trap "rm -rf $V7X_LOCAL" EXIT
     $L -I scripts/v7x/x30_vqascore_host.py rank --inputs /home/infres/yinwang/CS_QMI/outputs/V7/data/genai_rank_inputs.jsonl --out $X/b2_vqascore_qwen25vl7b_${1/\//of}.jsonl --shard $1 --limit ${2:-0}; rc=$?;;
  hpsv3-rank) H=/projects/EEG-foundation-model/yinghao/envs/v7x_hps/bin/python; X=$BIG/x30; mkdir -p $X; export V7X_LOCAL=/tmp/v7x_${SLURM_JOB_ID}; mkdir -p $V7X_LOCAL; trap "rm -rf $V7X_LOCAL" EXIT
     $H -I scripts/v7x/x30_hpsv3.py rank --inputs /home/infres/yinwang/CS_QMI/outputs/V7/data/genai_rank_inputs.jsonl --out $X/b5_hpsv3_${1/\//of}.jsonl --shard $1 --limit ${2:-0}; rc=$?;;
  pnvqa-gen) L=/projects/EEG-foundation-model/yinghao/envs/v7x_llm/bin/python; X=$BIG/x30; mkdir -p $X
     $L -I scripts/v7x/x30_pnvqa.py gen --prompts /home/infres/yinwang/CS_QMI/outputs/V7/data/genai_rank_prompts.jsonl --elements $2 --out $X/pnvqa_gen_${2}_${1/\//of}.jsonl --shard $1 --limit ${3:-0}; rc=$?;;
  pnvqa-score) L=/projects/EEG-foundation-model/yinghao/envs/v7x_llm/bin/python; X=$BIG/x30; mkdir -p $X; export V7X_LOCAL=/tmp/v7x_${SLURM_JOB_ID}; mkdir -p $V7X_LOCAL; trap "rm -rf $V7X_LOCAL" EXIT
     $L -I scripts/v7x/x30_pnvqa.py score --gen $X/pnvqa_gen_${2}_*.jsonl --inputs /home/infres/yinwang/CS_QMI/outputs/V7/data/genai_rank_inputs.jsonl --out $X/pnvqa_scores_${2}_${1/\//of}.jsonl --shard $1 --limit ${3:-0}; rc=$?;;
  dsg-gen) L=/projects/EEG-foundation-model/yinghao/envs/v7x_llm/bin/python; X=$BIG/x30; mkdir -p $X
     $L -I scripts/v7x/x30_dsg.py gen --prompts /home/infres/yinwang/CS_QMI/outputs/V7/data/genai_rank_prompts.jsonl --out $X/dsg_gen_${1/\//of}.jsonl --shard $1 --limit ${2:-0}; rc=$?;;
  dev-*) X=$BIG/x30_dev; mkdir -p $X; D=/home/infres/yinwang/CS_QMI/outputs/V7/data; IN=$D/evalmuse_dev_inputs.jsonl; PR=$D/evalmuse_dev_prompts.jsonl; IMG=$BIG/x30/evalmuse_dev_images
     T=/projects/EEG-foundation-model/yinghao/envs/v7_tifa/bin/python; L=/projects/EEG-foundation-model/yinghao/envs/v7x_llm/bin/python; H=/projects/EEG-foundation-model/yinghao/envs/v7x_hps/bin/python
     export HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf_cache_v7x MODELSCOPE_CACHE=$X/modelscope_cache V7X_LOCAL=/tmp/v7x_${SLURM_JOB_ID}; mkdir -p $V7X_LOCAL; trap "rm -rf $V7X_LOCAL" EXIT
     case "$what" in
       dev-tifa-qg) $T -I scripts/v7x/x30_tifa.py qg --prompts $PR --out $X/tifa_qa.jsonl; rc=$?;;
       dev-tifa-score) $T -I scripts/v7x/x30_tifa.py score --qa $X/tifa_qa.jsonl --inputs $IN --image-root $IMG --out $X/tifa_scores_${1/\//of}.jsonl --shard $1; rc=$?;;
       dev-dsg-gen) $L -I scripts/v7x/x30_dsg.py gen --prompts $PR --out $X/dsg_gen_${1/\//of}.jsonl --shard $1; rc=$?;;
       dev-dsg-score) $T -I scripts/v7x/x30_dsg.py score --dsg $X/dsg_gen_*.jsonl --inputs $IN --image-root $IMG --out $X/dsg_scores_${1/\//of}.jsonl --shard $1; rc=$?;;
       dev-pnvqa-gen) $L -I scripts/v7x/x30_pnvqa.py gen --prompts $PR --elements $2 --out $X/pnvqa_gen_${2}_${1/\//of}.jsonl --shard $1; rc=$?;;
       dev-pnvqa-score) $L -I scripts/v7x/x30_pnvqa.py score --gen $X/pnvqa_gen_${2}_*.jsonl --inputs $IN --image-root $IMG --out $X/pnvqa_scores_${2}_${1/\//of}.jsonl --shard $1; rc=$?;;
       dev-b2) $L -I scripts/v7x/x30_vqascore_host.py rank --inputs $IN --image-root $IMG --out $X/b2_vqascore_qwen25vl7b.jsonl; rc=$?;;
       dev-hpsv3) $H -I scripts/v7x/x30_hpsv3.py rank --inputs $IN --image-root $IMG --out $X/b5_hpsv3.jsonl; rc=$?;;
       dev-b1) B1W=$BIG/b1_work; mkdir -p $B1W; ln -sfn /projects/EEG-foundation-model/yinghao/models/hf_cache_v7/hub $B1W/hf_cache
          (cd $B1W && PATH=/projects/EEG-foundation-model/yinghao/envs/v7_vqascore/bin:$PATH HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf_cache_v7 /projects/EEG-foundation-model/yinghao/envs/v7_vqascore/bin/python -I \
             /home/infres/yinwang/CS_QMI/ssl_pilot_v7/scripts/v7/v7_vqascore.py rank --inputs $IN --images $IMG --out $X/b1_vqascore_clipflant5xxl.jsonl --batch 16); rc=$?;;
       *) echo "unknown dev mode $what"; rc=2;;
     esac;;
  *) echo "unknown mode $what"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
