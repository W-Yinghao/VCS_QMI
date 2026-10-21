#!/usr/bin/env python
"""Environment gate run at the start of every KD Slurm job, BEFORE any lock receipt is written.

1. `pip freeze` of env_kd must hash to ENV_LOCK.json (the overlay inherits the V9/V10 env; an in-place
   change there would otherwise silently change KD runs).
2. Key versions must equal the recorded ones.
3. With --cuda: a visible CUDA device must run a float32 matmul and a cuDNN conv forward/backward and agree
   with CPU float64 within loose tolerances (functional check, not a precision claim). Package precision
   settings are NOT changed here; the effective TF32 flags are only recorded.
Exit 0 = PASS, 1 = FAIL. A FAIL must stop the wrapper before training starts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "env" / "ENV_LOCK.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--out")
    args = p.parse_args()
    lock = json.loads(LOCK.read_text())
    result = {"schema": "kd_server_env_check_v1", "python": sys.executable, "checks": {}}
    ok = True
    # Both venvs symlink the same base interpreter, so only sys.prefix identifies env_kd.
    good = sys.prefix == lock["env_path"]
    result["checks"]["interpreter"] = "PASS" if good else f"FAIL: prefix {sys.prefix} != {lock['env_path']}"
    ok &= good
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout
    digest = hashlib.sha256(freeze.encode()).hexdigest()
    result["pip_freeze_sha256"] = digest
    result["checks"]["pip_freeze"] = "PASS" if digest == lock["pip_freeze_sha256"] else "FAIL: differs from ENV_LOCK"
    ok &= digest == lock["pip_freeze_sha256"]
    import platform
    import numpy
    import torch
    import torchvision
    seen = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "numpy": numpy.__version__}
    for key, value in seen.items():
        expected = lock["key_versions"][key]
        good = value == expected
        result["checks"]["version_" + key] = "PASS" if good else f"FAIL: {value} != {expected}"
        ok &= good
    result["versions"] = seen
    if args.cuda:
        try:
            if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
                raise RuntimeError("no visible CUDA device")
            dev = torch.device("cuda")
            props = torch.cuda.get_device_properties(dev)
            result["gpu"] = {"name": props.name, "capability": [props.major, props.minor],
                             "total_memory": props.total_memory, "device_count": torch.cuda.device_count()}
            g = torch.Generator().manual_seed(1)
            a, b = torch.randn(256, 256, generator=g), torch.randn(256, 256, generator=g)
            mm = (a.to(dev) @ b.to(dev)).cpu().double()
            ref = a.double() @ b.double()
            mm_rel = float((mm - ref).norm() / ref.norm())
            x = torch.randn(8, 3, 32, 32, generator=g)
            conv = torch.nn.Conv2d(3, 16, 3, padding=1)
            ref_out = torch.nn.functional.conv2d(x.double(), conv.weight.double(), conv.bias.double(), padding=1)
            conv_d = conv.to(dev)
            xd = x.to(dev).requires_grad_(True)
            out = conv_d(xd)
            out.square().mean().backward()
            torch.cuda.synchronize(dev)
            conv_rel = float((out.detach().cpu().double() - ref_out).norm() / ref_out.norm())
            finite = bool(torch.isfinite(xd.grad).all()) and bool(torch.isfinite(conv_d.weight.grad).all())
            # TF32 may be active for cuDNN convolutions by torch default; tolerances are functional only.
            good = mm_rel < 1e-4 and conv_rel < 5e-3 and finite
            result["checks"]["cuda_functional"] = ("PASS" if good else "FAIL") + f": matmul_rel={mm_rel:.2e} conv_rel={conv_rel:.2e} grad_finite={finite}"
            result["effective_precision_flags"] = {"matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                                                   "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                                                   "float32_matmul_precision": torch.get_float32_matmul_precision()}
            ok &= good
        except Exception as exc:  # any CUDA problem must stop the job before a receipt exists
            result["checks"]["cuda_functional"] = f"FAIL: {type(exc).__name__}: {exc}"
            ok = False
    result["status"] = "PASS" if ok else "FAIL"
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text, flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
