#!/usr/bin/env bash
#SBATCH --job-name=v7c_fidelity_gen
#SBATCH --partition=RTX6000PRO,H100
#SBATCH --exclude=node51,node60
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V7C-03 source fidelity (label-free): the frozen local question generators (B4T TIFA released LLaMA-2 route; B4D DSG with Qwen2.5-32B) on the
# fixed fidelity prompt sets (scripts/v7x/v7c_fidelity_prep.py), for comparison with the authors' released question files.  Same runners and
# environments as the frozen x30 TIFA / DSG generation modes.
set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7; export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
BIG=/projects/EEG-foundation-model/yinghao/outputs_v7; F=/home/infres/yinwang/CS_QMI/outputs/V7_CLOSEOUT/fidelity
export TORCH_HOME=/projects/EEG-foundation-model/yinghao/models/torch_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD)"
T=/projects/EEG-foundation-model/yinghao/envs/v7_tifa/bin/python; L=/projects/EEG-foundation-model/yinghao/envs/v7x_llm/bin/python
export HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf_cache_v7x MODELSCOPE_CACHE=$BIG/x30/modelscope_cache
export V7X_LOCAL=/tmp/v7x_${SLURM_JOB_ID}; mkdir -p $V7X_LOCAL; trap "rm -rf $V7X_LOCAL" EXIT
t0=$(date +%s); $T -I scripts/v7x/x30_tifa.py qg --prompts $F/tifa_v1_50_prompts.jsonl --out $F/local_tifa_qa.jsonl; r1=$?; t1=$(date +%s)
$L -I scripts/v7x/x30_dsg.py gen --prompts $F/dsg1k_50_prompts.jsonl --out $F/local_dsg_gen.jsonl --shard 0/1; r2=$?; t2=$(date +%s)
echo "tifa_qg_rc=$r1 seconds=$((t1 - t0)) dsg_gen_rc=$r2 seconds=$((t2 - t1)) end=$(date -u +%FT%TZ)"; exit $(( r1 | r2 ))
