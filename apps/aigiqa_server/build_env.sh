#!/usr/bin/env bash
# Build the AIGIQA env (package INSTALL_CN.md section 4 + aigiqa/README_CN.md section 1), login node.
# Usage: bash build_env.sh <scratch_dir>   (pip/conda caches and TMPDIR go to the node-local scratch dir, not home)
# Decision record: Python 3.11 (package: 3.10/3.11 for the old BLIP stack); torch 2.9.0+cu128 / torchvision 0.24.0+cu128,
# the CUDA 12.8 build the project env already runs on RTX PRO 6000 Blackwell (sm_120), L40S and H100; cu12.4-and-older
# wheels have no sm_120 kernels. The BLIP stack pins are the package's own (pure Python, no torch-ABI extensions).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/paths.env"
SCRATCH=${1:?scratch dir}
mkdir -p "$SCRATCH/pip_cache" "$SCRATCH/tmp" "$SCRATCH/conda_pkgs"
export PIP_CACHE_DIR=$SCRATCH/pip_cache TMPDIR=$SCRATCH/tmp CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs
export PIP_NO_INPUT=1 PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
[ -e "$AIG_ENV" ] && { echo "refuse: $AIG_ENV exists" >&2; exit 1; }
/home/infres/yinwang/anaconda3/bin/conda create -y -q -p "$AIG_ENV" python=3.11 pip setuptools wheel
PIP=("$AIG_PY" -m pip install --upgrade-strategy only-if-needed)
"${PIP[@]}" --index-url https://download.pytorch.org/whl/cu128 "torch==2.9.0+cu128" "torchvision==0.24.0+cu128"
# package README section 1, in its order; -c keeps torch / numpy where they are
"${PIP[@]}" -c "$HERE/constraints.txt" "numpy==1.26.4" "scipy==1.13.1" "pyarrow==14.0.2" pillow einops ftfy regex tqdm "PyYAML>=6,<7"
"${PIP[@]}" -c "$HERE/constraints.txt" "timm==0.6.13" "transformers==4.30.2" "fairscale==0.4.13" "huggingface_hub==0.23.5"
"${PIP[@]}" -c "$HERE/constraints.txt" "diffusers==0.19.3" "accelerate==0.20.3" "datasets==2.14.7"
# runtime pkg_resources for accelerate 0.20.3 / diffusers 0.19.3 (removed from setuptools>=81); see constraints.txt
"${PIP[@]}" -c "$HERE/constraints.txt" "setuptools==69.5.1"
"${PIP[@]}" --no-deps "git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
# ImageReward's setup.py imports pkg_resources, which setuptools>=81 no longer ships: constrain only its isolated build env
"${PIP[@]}" --no-deps --build-constraint "$HERE/build_constraints.txt" "git+https://github.com/THUDM/ImageReward.git@78f8051059b34fef8e0032ccb40561c23a2ddbbb"
"$AIG_PY" -m pip check || echo "pip check reported conflicts (recorded in ENV_LOCK.json)"
"$AIG_PY" -m pip freeze --all > "$HERE/evidence/env/pip_freeze.txt"
echo "built $AIG_ENV"
