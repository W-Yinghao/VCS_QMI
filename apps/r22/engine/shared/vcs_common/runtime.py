"""Small provenance and resource helpers; no TF32/precision overrides."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, value) -> None:
    """Atomic complete-file publication, preserving failures as failures."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def seed_all(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtime_record() -> dict:
    import numpy as np
    import torch
    record = dict(timestamp_utc=utc_now(), python=sys.version, platform=platform.platform(),
                  torch=torch.__version__, numpy=np.__version__, cuda_build=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(), cudnn_version=torch.backends.cudnn.version(),
                  float32_matmul_precision=torch.get_float32_matmul_precision(),
                  matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                  cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  cudnn_benchmark=torch.backends.cudnn.benchmark,
                  threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                  precision_policy="retain_torch_defaults_record_effective_settings_no_AMP_in_dependence")
    record["environment_overrides"] = {name: os.environ.get(name) for name in (
        "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF")}
    record["gpus"] = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        record["gpus"].append(dict(index=i, name=p.name, total_memory=p.total_memory,
                                   capability=[p.major, p.minor]))
    if torch.cuda.is_available():
        try:
            record["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, timeout=10).strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            record["driver"] = None
    return record


def synchronize(device) -> None:
    import torch
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class TimedRegion:
    """Synchronized end-to-end duration for a supplied region, never FLOP proxy."""
    def __init__(self, device="cpu"):
        self.device = device
        self.seconds = None

    def __enter__(self):
        synchronize(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        synchronize(self.device)
        self.seconds = time.perf_counter() - self.start

