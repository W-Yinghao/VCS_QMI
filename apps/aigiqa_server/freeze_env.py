"""Write or check the AIGIQA environment lock (package INSTALL_CN.md: save pip freeze, Python/torch/CUDA/cuDNN, then freeze).

  write --lock ENV_LOCK.json   login node, right after build_env.sh (no GPU needed)
  check --lock ENV_LOCK.json   first step of every AIGIQA job; exit 3 on any drift, so no run starts in a changed env
The GPU-side facts (device, driver, effective TF32 flags) are recorded per job by the job itself (G0 / run records).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import subprocess
import sys
import time
from pathlib import Path

PINNED_SOURCES = {"clip": "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6", "image-reward": "78f8051059b34fef8e0032ccb40561c23a2ddbbb"}
KEY_DISTS = ["torch", "torchvision", "numpy", "scipy", "pyarrow", "Pillow", "PyYAML", "timm", "transformers", "tokenizers", "fairscale",
             "huggingface_hub", "diffusers", "accelerate", "datasets", "safetensors", "clip", "image-reward", "einops", "ftfy", "regex", "tqdm"]


def snapshot():
    import torch
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True).stdout.splitlines()
    freeze = sorted(line for line in freeze if line.strip())
    direct = {}
    for name in PINNED_SOURCES:
        try:
            text = md.distribution(name).read_text("direct_url.json")
            direct[name] = json.loads(text) if text else None
        except md.PackageNotFoundError:
            direct[name] = None
    versions = {}
    for name in KEY_DISTS:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return {"python_executable": sys.executable, "python_version": sys.version,
            "pip_freeze_all": freeze, "pip_freeze_sha256": hashlib.sha256("\n".join(freeze).encode()).hexdigest(),
            "key_versions": versions, "direct_url": direct,
            "torch": {"version": torch.__version__, "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                      "arch_flags": torch._C._cuda_getArchFlags(), "git_version": torch.version.git_version,
                      "default_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "default_cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                      "float32_matmul_precision": torch.get_float32_matmul_precision()}}


def pinned_ok(snap):
    bad = {}
    for name, commit in PINNED_SOURCES.items():
        info = snap["direct_url"].get(name) or {}
        got = (info.get("vcs_info") or {}).get("commit_id")
        if got != commit:
            bad[name] = got
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["write", "check"])
    ap.add_argument("--lock", required=True)
    a = ap.parse_args()
    snap = snapshot()
    bad = pinned_ok(snap)
    if a.action == "write":
        if Path(a.lock).exists():
            raise SystemExit(f"refuse to overwrite {a.lock}")
        if bad:
            raise SystemExit(f"pinned source commits differ: {bad}")
        body = {"schema": "aigiqa_server_env_lock_v1", "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **snap}
        Path(a.lock).write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"written": a.lock, "pip_freeze_sha256": snap["pip_freeze_sha256"], "torch": snap["torch"]["version"]}, indent=1))
        return
    lock = json.loads(Path(a.lock).read_text())
    drift = {}
    for key in ("python_executable", "python_version", "pip_freeze_sha256"):
        if snap[key] != lock[key]:
            drift[key] = {"lock": lock[key], "live": snap[key]}
    for key in ("version", "cuda_build", "cudnn", "arch_flags"):
        if snap["torch"][key] != lock["torch"][key]:
            drift["torch." + key] = {"lock": lock["torch"][key], "live": snap["torch"][key]}
    if bad:
        drift["pinned_sources"] = bad
    if drift:
        changed = sorted(set(snap["pip_freeze_all"]) ^ set(lock["pip_freeze_all"]))
        print(json.dumps({"status": "ENV_DRIFT", "drift": drift, "freeze_diff": changed[:50]}, indent=1, default=str))
        raise SystemExit(3)
    print(json.dumps({"status": "ENV_MATCHES_LOCK", "pip_freeze_sha256": snap["pip_freeze_sha256"]}))


if __name__ == "__main__":
    main()
