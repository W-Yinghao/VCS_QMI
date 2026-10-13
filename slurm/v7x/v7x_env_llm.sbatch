#!/usr/bin/env bash
#SBATCH --job-name=v7x_env_llm
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# envs/v7x_llm for the LLM / MLLM side of V7X-30: DSG and PN-VQA generation (Qwen2.5-32B-Instruct), PN-VQA VQA (Qwen2-VL-7B-Instruct), B2 VQAScore host
# (t2v-metrics 3.0 qwen2.5-vl-7b, scoped import).  torch 2.9.0 cu128 (RTX6000PRO / H100 / L40S), transformers 4.57.6 (as vl_baselines, which is left unchanged).
set -eo pipefail; export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
E=/projects/EEG-foundation-model/yinghao/envs/v7x_llm; CONDA=/home/infres/yinwang/anaconda3/bin/conda
[ -x $E/bin/python ] || $CONDA create -y -q -p $E python=3.10
$E/bin/pip install -q torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
$E/bin/pip install -q "transformers==4.57.6" "accelerate>=1.0" sentencepiece protobuf "numpy<2.3" pandas scipy tqdm Pillow "openai>=1.0" qwen-vl-utils decord einops
$E/bin/pip install -q --no-deps "t2v-metrics==3.0"
$E/bin/pip check || echo "pip check reported conflicts (recorded above)"
$E/bin/python -I -c "import torch, transformers, pandas; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'pandas', pandas.__version__)"
$E/bin/pip freeze > /home/infres/yinwang/CS_QMI/outputs/V7/env_v7x_llm_freeze.txt; echo "exit=0 end=$(date -u +%FT%TZ)"
