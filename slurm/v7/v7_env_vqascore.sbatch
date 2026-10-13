#!/usr/bin/env bash
#SBATCH --job-name=v7_env_vqascore
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V7 baseline B1 env: t2v-metrics==3.0 (the paper-reproduction release, wheel sha256 448398dc...), its pins torch 2.5.1 / torchvision 0.20.1 /
# transformers 4.49.0 / numpy 1.26.4 (cu124 wheels: L40S / H100; not Blackwell).  The release is installed --no-deps, then its declared
# dependencies minus those that cannot install or are not on the CLIP-FlanT5 path (apex 0.9.10dev, deepspeed, bitsandbytes, xformers, torchcodec,
# wandb, google-genai); the import test below is the check.
set -x; date -u; export PYTHONNOUSERSITE=1
E=/projects/EEG-foundation-model/yinghao/envs/v7_vqascore
[ -x $E/bin/python ] || ~/anaconda3/bin/conda create -y -q -p $E python=3.10 || exit 1
$E/bin/pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 || exit 2
$E/bin/pip install -q --no-deps t2v-metrics==3.0 || exit 3
$E/bin/pip install -q "numpy==1.26.4" "transformers==4.49.0" sentencepiece "ftfy>=6.1.1" tqdm "huggingface-hub>=0.19.4" matplotlib "open-clip-torch>=2.23.0" \
  opencv-python-headless pandas scipy datasets tokenizers omegaconf iopath fairscale scikit-learn pycocoevalcap hpsv2 "fire==0.4.0" tiktoken "peft==0.5.0" \
  decord easydict protobuf pytz av icecream markdown2 pydantic accelerate shortuuid requests "httpx==0.24.0" einops-exts einops PyYAML moviepy \
  "fvcore==0.1.5.post20221221" "imageio==2.31.1" "Pillow==10.0.0" "psutil==5.9.5" "termcolor==2.4.0" qwen-vl-utils func_timeout "timm>=1.0.15" gitpython \
  blobfile diffusers nltk rouge-score openai || exit 4
$E/bin/pip check || echo "pip check reported conflicts (logged above)"
$E/bin/pip freeze > $E/FREEZE.txt
cd /tmp && $E/bin/python -I - <<'PY'
import torch, transformers, numpy
print("torch", torch.__version__, "| transformers", transformers.__version__, "| numpy", numpy.__version__)
import t2v_metrics
from t2v_metrics.models.vqascore_models.clip_t5_model import CLIPT5Model, CLIP_T5_MODELS, default_question_template, default_answer_template
print("t2v_metrics import ok:", t2v_metrics.__file__); print("clip-flant5-xxl:", CLIP_T5_MODELS["clip-flant5-xxl"]); print("templates:", default_question_template, "|", default_answer_template)
PY
rc=$?; date -u; echo "ENV_DONE rc=$rc"; exit $rc
