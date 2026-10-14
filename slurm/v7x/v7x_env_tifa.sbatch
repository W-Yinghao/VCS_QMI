#!/usr/bin/env bash
#SBATCH --job-name=v7x_env_tifa
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# envs/v7_tifa for the B4T (TIFA) / B4D (DSG) VQA side: torch 2.5.1 cu124 (L40S / H100; no Blackwell kernels), transformers 4.40.2, ModelScope (mPLUG-large).
# The official tifa@8e37050 / DSG@7330119 trees stay read-only in third_party and are imported by path (scoped import; nothing pip-installed from them).
set -eo pipefail; export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
E=/projects/EEG-foundation-model/yinghao/envs/v7_tifa; CONDA=/home/infres/yinwang/anaconda3/bin/conda
[ -x $E/bin/python ] || $CONDA create -y -q -p $E python=3.10
$E/bin/pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
$E/bin/pip install -q "transformers==4.40.2" "accelerate==0.30.1" sentencepiece protobuf "numpy<2" pandas scipy tqdm Pillow word2number "openai>=1.0" \
   "modelscope==1.20.1" timm opencv-python-headless addict simplejson sortedcontainers oss2 "datasets<3" rapidfuzz einops ftfy regex "setuptools<70" decord librosa soundfile zhconv unicodedata2
$E/bin/pip install -q torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
$E/bin/pip install -q "fairseq==0.12.2" || echo "fairseq 0.12.2 did not install: the x_stubs fallback is used (recorded)"
$E/bin/pip check || echo "pip check reported conflicts (recorded above)"
cd /home/infres/yinwang/CS_QMI/ssl_pilot_v7 && $E/bin/python -I -c "import sys; sys.path.insert(0, 'scripts/v7x'); import x_stubs; x_stubs.install(tifa=True); import pkg_resources, torch, transformers, modelscope; from modelscope.pipelines import pipeline; from modelscope.preprocessors.multi_modal import OfaPreprocessor, MPlugPreprocessor; print('torch', torch.__version__, 'transformers', transformers.__version__, 'modelscope', modelscope.__version__)"
$E/bin/pip freeze > /home/infres/yinwang/CS_QMI/outputs/V7/env_v7_tifa_freeze.txt; echo "exit=0 end=$(date -u +%FT%TZ)"
