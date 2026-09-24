"""Small provenance / IO helpers shared by all CLIs."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def md5_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default)


def _json_default(o: Any):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, float) and (o != o):
        return None
    raise TypeError(f"not JSON serializable: {type(o)}")


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def state_dict_sha256(module: torch.nn.Module) -> str:
    """Hash of all tensors (parameters and buffers) in state_dict order."""
    h = hashlib.sha256()
    for k, v in module.state_dict().items():
        h.update(k.encode("utf-8"))
        t = v.detach().cpu().contiguous()
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: str | Path, obj: Any, indent: int | None = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, ensure_ascii=False, default=_json_default) + "\n")


def append_jsonl(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_info(repo_root: str | Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=str(repo_root), capture_output=True, text=True, timeout=20)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:  # noqa: BLE001 - provenance only
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "repo_root": str(repo_root),
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": status.splitlines() if status else [],
        "is_dirty": bool(status),
    }


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_count_visible": os.cpu_count(),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
    }
    try:
        import torchvision  # noqa: PLC0415

        info["torchvision"] = torchvision.__version__
    except Exception as e:  # noqa: BLE001
        info["torchvision"] = f"import failed: {e!r}"
    try:
        import numpy  # noqa: PLC0415

        info["numpy"] = numpy.__version__
    except Exception as e:  # noqa: BLE001
        info["numpy"] = f"import failed: {e!r}"
    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({"index": i, "name": p.name, "total_memory_mb": int(p.total_memory // (1024 * 1024)),
                         "capability": f"{p.major}.{p.minor}", "multi_processor_count": p.multi_processor_count})
        info["gpus"] = gpus
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20)
            info["nvidia_smi"] = out.stdout.strip().splitlines() if out.returncode == 0 else out.stderr.strip()
        except Exception as e:  # noqa: BLE001
            info["nvidia_smi"] = f"unavailable: {e!r}"
    return info


def precision_flags() -> dict[str, Any]:
    return {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def apply_precision_policy(train_cfg: dict[str, Any]) -> None:
    """FP32 single-GPU reference: TF32 off, cudnn.benchmark off, no compile."""
    if train_cfg["precision"] != "fp32":
        raise ValueError("first round is FP32 only")
    torch.backends.cudnn.benchmark = False
    if not train_cfg["allow_tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    if train_cfg["compile"]:
        raise ValueError("torch.compile is disabled in the first round")


class Timer:
    """Wall-clock timer that synchronizes CUDA when a device is given."""

    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.t0 = 0.0

    def __enter__(self):
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self.t0
        return False
