#!/usr/bin/env bash
#SBATCH --job-name=v7x_env_hps
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# envs/v7x_hps for B5 HPSv3 (pip hpsv3 1.0.0, which pins transformers 4.45.2): torch 2.9.0 cu128 (RTX6000PRO / H100 / L40S); flash-attn not installed
# (the runner passes a config copy with disable_flash_attn2: true).
set -eo pipefail; export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
E=/projects/EEG-foundation-model/yinghao/envs/v7x_hps; CONDA=/home/infres/yinwang/anaconda3/bin/conda
[ -x $E/bin/python ] || $CONDA create -y -q -p $E python=3.10
$E/bin/pip install -q torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
$E/bin/pip install -q "transformers==4.45.2" "accelerate>=0.30" "peft>=0.8,<0.14" "trl==0.11.4" fire "omegaconf>=2.3" "datasets<3" safetensors einops qwen-vl-utils timm "numpy<2" opencv-python-headless pyyaml tqdm Pillow matplotlib pandas requests tensorboard
$E/bin/pip install -q --no-deps hpsv3==1.0.0
$E/bin/pip check || echo "pip check reported conflicts (recorded above)"
$E/bin/python -I -c "import torch, transformers, hpsv3; from hpsv3 import HPSv3RewardInferencer; print('torch', torch.__version__, 'transformers', transformers.__version__)"
$E/bin/pip freeze > /home/infres/yinwang/CS_QMI/outputs/V7/env_v7x_hps_freeze.txt; echo "exit=0 end=$(date -u +%FT%TZ)"
