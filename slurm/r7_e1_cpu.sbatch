#!/usr/bin/env bash
#SBATCH --job-name=r7_e1_cpu
#SBATCH --partition=CPU
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/infres/yinwang/CS_QMI/slurm_logs/%x_%j.out
# r7 E1 CPU steps (external Qwen2.5-VL-7B baselines MMTok / HiPrune as a separate host column; reports/R7/R7_E1_EXTERNAL_PROTOCOL_DRAFT_20261013.yaml).
# First arg: env <mmtok|hiprune> (create the method's pinned env + import check; refuses an existing env dir; never on the login node)
# | importcheck <mmtok|hiprune> | selfcheck <mmtok|hiprune> (tiny random Qwen2.5-VL on CPU through the method's own code and the E1 glue)
# | tests (main env: stubbed unit tests) | plan [anchor] (configs + unit lines, nothing submitted) | complete [anchor] (stage 2: coverage + markers).
# Positional arguments only (no --export).  The fetched official code is read-only and is never the working directory; python -I everywhere.
source /home/infres/yinwang/CS_QMI/ssl_pilot/slurm/common.sh; set -o pipefail; cd /home/infres/yinwang/CS_QMI/ssl_pilot
ENVS=/projects/EEG-foundation-model/yinghao/envs; TP=/projects/EEG-foundation-model/yinghao/third_party
MMTOK_DIR=$TP/MMTok@8e26868; HIPRUNE_DIR=$TP/HiPrune@8278100
PY312=/home/infres/yinwang/CS_QMI/.uv_python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12   # the CPython 3.12 of the vl_baselines venv (both repos: python 3.12)
E1=/home/infres/yinwang/CS_QMI/outputs/R7_D1/r7_40/eval_e1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/projects/EEG-foundation-model/yinghao/models/hf PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
echo "host=$(hostname) start=$(date -u +%FT%TZ) commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain scripts slurm tests | wc -l)"
echo "code: $(cat scripts/r7_e1_external.py tests/test_r7_e1_external.py slurm/r7_e1_* 2>/dev/null | sha256sum | cut -c1-16) (sha256 of the E1 script, tests and wrappers)"
what="$1"; shift
method_dir() { case "$1" in mmtok) echo "$MMTOK_DIR";; hiprune) echo "$HIPRUNE_DIR";; *) echo "";; esac; }

build_env() {   # $1 = mmtok | hiprune, $2 = env dir, $3 = temp dir.  Pins from the official repos; what they leave unpinned resolved as of the method's last code commit.
  local m="$1" E="$2" T="$3"
  set -x
  "$PY312" -I -m venv "$E" || return 10
  "$E/bin/python" -I -m pip install -q --no-cache-dir pip==26.2.1 || return 11            # the installer only (needed for --uploaded-prior-to)
  if [ "$m" = mmtok ]; then
    # install.md: python 3.12; torch 2.8.0 / torchvision 0.23.0 / torchaudio 2.8.0 from the cu128 index; requirements.txt (transformers==4.52.4,
    # the rest unpinned -> as of the release commit 5b186b6, 2026-02-20T14:24:18Z); 'pip install -e .' replaced by importing the read-only tree
    # (the egg-info would be written into it); LLaVA-NeXT, lmms-eval, qwen-vl-utils, flash-attn: optional in install.md, not used on the Qwen path.
    "$E/bin/python" -I -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 || return 12
    printf 'torch==2.8.0\ntorchvision==0.23.0\ntorchaudio==2.8.0\n' > "$T/constraints.txt"
    "$E/bin/python" -I -m pip install --no-cache-dir --uploaded-prior-to 2026-02-20T14:24:18Z -c "$T/constraints.txt" \
        transformers==4.52.4 loguru sentencepiece protobuf accelerate pillow einops requests open_clip_torch || return 13
  else
    # setup_qwen.sh: transformers==4.52.0, accelerate, flash_attn ("FlashAttention is required"); python 3.12 (setup_llava.sh); torch unpinned ->
    # 2.8.0 (cu128), the last torch release before the last Qwen-code commit 716f2d5 (2025-09-24T15:29:24Z); torchvision >= 0.16 and numpy==1.26.4
    # from the bundled lmms-eval's dependency list; lmms-eval itself, qwen_vl_utils, av, openpyxl: evaluation-harness packages, not installed.
    "$E/bin/python" -I -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 || return 12
    printf 'torch==2.8.0\ntorchvision==0.23.0\n' > "$T/constraints.txt"
    "$E/bin/python" -I -m pip install --no-cache-dir --uploaded-prior-to 2025-09-24T15:29:24Z -c "$T/constraints.txt" \
        transformers==4.52.0 accelerate numpy==1.26.4 pillow einops || return 13
    local abi; abi=$("$E/bin/python" -I -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") || return 14
    # flash-attn 2.8.3 = the last release before 2025-09-24 (2025-08-14): its official prebuilt wheel for torch 2.8 / CUDA 12 / cp312 / this C++ ABI
    "$E/bin/python" -I -m pip install --no-cache-dir --no-deps \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abi${abi}-cp312-cp312-linux_x86_64.whl" || return 15
  fi
  set +x
  "$E/bin/python" -I -m pip freeze > "$E/FREEZE.txt" || return 16
}

make_env() {   # $1 = mmtok | hiprune: refuses an existing env dir; a failed build removes only the directory this job created
  local m="$1" E="$ENVS/$1" D T rc; D=$(method_dir "$1")
  [ -n "$D" ] || { echo "env: method must be mmtok | hiprune"; return 2; }
  [ -e "$E" ] && { echo "env: $E exists - not touched (delete it deliberately to rebuild)"; return 3; }
  [ -f "$D/LICENSE" ] || { echo "env: no LICENSE in $D - stop"; return 4; }
  local free_kb; free_kb=$(df -Pk /projects/EEG-foundation-model | awk 'NR==2 {print $4}')
  [ "$free_kb" -gt 1342177280 ] || { echo "env: /projects free ${free_kb} kB < 1.25 TB - stop (keep >= 1.2 TB)"; return 5; }
  curl -s -o /dev/null -m 30 https://pypi.org/simple/pip/ || { echo "env: no network on $(hostname) - stop"; return 6; }
  T=$(mktemp -d "$ENVS/_tmp_e1_${m}_${SLURM_JOB_ID:-x}_XXXX"); export TMPDIR=$T          # pip temp space on /projects (removed below)
  build_env "$m" "$E" "$T"; rc=$?; set +x; rm -rf "$T"; unset TMPDIR
  if [ $rc -ne 0 ]; then echo "env: build step failed (rc $rc); removing the partial $E created by this job"; rm -rf "$E"; return $rc; fi
  echo "env: $E built ($(du -sh "$E" | cut -f1)); free on /projects now $(df -h --output=avail /projects/EEG-foundation-model | tail -1)"; import_check "$m"
}

import_check() {   # the method's modules load in its env (cwd = this repo, never the fetched tree; the tree is appended to sys.path, as the official harnesses import it)
  local m="$1" E="$ENVS/$1" D; D=$(method_dir "$1")
  [ -x "$E/bin/python" ] || { echo "importcheck: no env $E"; return 2; }
  if [ "$m" = mmtok ]; then
    "$E/bin/python" -I -c '
import importlib.metadata as md, json, sys
sys.path.append(sys.argv[1])
import torch, transformers
import mmtok
from mmtok.qwen import mmtok_qwen2_5_vl
from mmtok.qwen.qwen2_5_VLmodel_mmtok import Qwen2_5_VL_MMTok
from mmtok.qwen.modeling_qwen2_5_vl_mmtok import Qwen2_5_VisionTransformerPretrainedModel_MMTok
from mmtok.core import MMTokCore
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
def ver(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None
v = {p: ver(p) for p in ("torch", "torchvision", "torchaudio", "transformers", "accelerate", "tokenizers", "numpy", "pillow", "loguru", "flash_attn")}
print("IMPORTCHECK", json.dumps({"method": "mmtok", "ok": True, "python": sys.version.split()[0], "cuda_build": torch.version.cuda, "versions": v,
      "mmtok_version": mmtok.__version__, "mmtok_file": mmtok.__file__, "wrapper": mmtok_qwen2_5_vl.__module__}))' "$D"
  else
    HIPRUNE_QWEN_RETENTION=0.25 HIPRUNE_ALPHA=0.1 HIPRUNE_OBJECT_LAYER=16 "$E/bin/python" -I -c '
import importlib.metadata as md, inspect, json, sys
sys.path.append(sys.argv[1])
import torch, transformers
import transformers.modeling_flash_attention_utils as FAU
src = inspect.getsource(FAU)
fa = {}
try:
    import flash_attn
    from flash_attn import flash_attn_varlen_func  # noqa: F401
    from flash_attn.layers.rotary import apply_rotary_emb  # noqa: F401
    fa = {"flash_attn_import": True, "flash_attn_version": flash_attn.__version__}
except Exception as e:
    fa = {"flash_attn_import": False, "error": f"{type(e).__name__}: {e}"}
from Qwen2_5_VL.qwen2_5_vl_HiPrune import Qwen2_5_VLForConditionalGeneration
import Qwen2_5_VL.qwen2_5_vl_HiPrune as HP
def ver(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None
v = {p: ver(p) for p in ("torch", "torchvision", "transformers", "accelerate", "tokenizers", "numpy", "pillow", "flash_attn")}
print("IMPORTCHECK", json.dumps({"method": "hiprune", "ok": True, "python": sys.version.split()[0], "cuda_build": torch.version.cuda, "versions": v,
      "module_file": HP.__file__, "knobs_read_at_import": {"RETAIN": HP.RETAIN, "ALPHA": HP.ALPHA, "OBJECT_LAYER": HP.OBJECT_LAYER},
      "fa2_available_here": bool(FAU.is_flash_attn_available()), "guarded_names_in_transformers_fa_utils": {n: (n in src) for n in ("apply_rotary_emb", "flash_attn_varlen_func", "_flash_attention_forward")},
      **fa}))' "$D"
  fi
}

case "$what" in
  env) make_env "$1"; rc=$?;;
  importcheck) import_check "$1"; rc=$?;;
  selfcheck)   # tiny random Qwen2.5-VL (CPU) through the method's code and the E1 worker: pruning runs, kept == the method's own k rule
    E="$ENVS/$1"; [ -x "$E/bin/python" ] || { echo "selfcheck: no env $E"; exit 2; }
    "$E/bin/python" -I scripts/r7_e1_external.py selfcheck --method "$1" --out "$E1/checks"; rc=$?;;
  tests) "$PYTHON" -I tests/test_r7_e1_external.py -v; rc=$?;;
  plan) "$PYTHON" -I scripts/r7_e1_external.py plan --root "$E1" --lines "$E1/r7_e1_lines.txt" ${1:+--anchor}; rc=$?;;
  complete) "$PYTHON" -I scripts/r7_e1_external.py complete --pred-dir "$E1/pred" --out "$E1" ${1:+--anchor}; rc=$?;;
  *) echo "usage: env <mmtok|hiprune> | importcheck <mmtok|hiprune> | selfcheck <mmtok|hiprune> | tests | plan [anchor] | complete [anchor]"; rc=2;;
esac
echo "exit=$rc end=$(date -u +%FT%TZ)"; exit $rc
