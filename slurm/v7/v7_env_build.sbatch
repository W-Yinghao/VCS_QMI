#!/usr/bin/env bash
#SBATCH --job-name=v7_env_build
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# V7 FGA-BLIP2 environment (reports/V7/V7_00_INTAKE.md): python 3.10, torch 2.9.0 cu128 (L40S / H100 / RTX PRO 6000 Blackwell), and the
# EvalMuse@45c8bb4 pins that its trimmed LAVIS import path needs (transformers 4.33.2, timm 0.4.12, fairscale 0.4.4).  Packages the trimmed
# fork never imports (diffusers, open3d, spacy, decord, streamlit, moviepy, ...) are left out; the import test below is the check.
set -x; date -u
export PYTHONNOUSERSITE=1   # pip must not count ~/.local packages (e.g. nvidia-cufile-cu12) as installed: the env runs with python -I
E=/projects/EEG-foundation-model/yinghao/envs/v7_fga; FGA=/projects/EEG-foundation-model/yinghao/third_party/EvalMuse@45c8bb4
[ -x $E/bin/python ] || ~/anaconda3/bin/conda create -y -q -p $E python=3.10 || exit 1
$E/bin/pip install -q torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128 || exit 2
$E/bin/pip install -q "numpy<2" transformers==4.33.2 timm==0.4.12 fairscale==0.4.4 omegaconf iopath webdataset opencv-python-headless==4.10.0.84 \
  decord einops ftfy contexttimer pandas scipy scikit-learn pycocotools pycocoevalcap sentencepiece easydict==1.9 h5py pyarrow pillow pyyaml pytest || exit 3
$E/bin/pip check || echo "pip check reported conflicts (logged above)"
$E/bin/pip freeze > $E/FREEZE.txt
cd /tmp && PYTHONDONTWRITEBYTECODE=1 $E/bin/python -I - "$FGA" <<'PY'
import sys, importlib
sys.path.insert(0, sys.argv[1])
import torch, torchvision, transformers, timm, numpy
print("torch", torch.__version__, "cuda build", torch.version.cuda, "| torchvision", torchvision.__version__, "| transformers", transformers.__version__,
      "| timm", timm.__version__, "| numpy", numpy.__version__)
import lavis
from lavis.models import load_model_and_preprocess
from lavis.models.blip2_models.fga_blip2 import FGA_Blip2
from lavis.processors.blip_processors import BlipImageEvalProcessor, Blip2ImageTrainProcessor, BlipCaptionProcessor
from lavis.datasets.datasets.align_datasets import AlignmentDataset
from lavis.runners.runner_base import RunnerBase
import lavis.tasks
print("lavis import ok:", FGA_Blip2.__name__, RunnerBase.__name__)
PY
rc=$?; date -u; echo "ENV_DONE rc=$rc"; exit $rc
