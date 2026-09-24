"""Atomic checkpoints with full RNG capture (spec §13)."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_rng(device: torch.device | None, loader_gen: torch.Generator | None, pair_gen: torch.Generator | None) -> dict[str, Any]:
    st: dict[str, Any] = {
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch_cpu": torch.get_rng_state(),
        "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_loader_generator": loader_gen.get_state() if loader_gen is not None else None,
        "rng_pair_generator": pair_gen.get_state() if pair_gen is not None else None,
    }
    return st


def restore_rng(st: dict[str, Any], loader_gen: torch.Generator | None, pair_gen: torch.Generator | None) -> None:
    random.setstate(st["rng_python"])
    np.random.set_state(st["rng_numpy"])
    torch.set_rng_state(st["rng_torch_cpu"])
    if st.get("rng_torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["rng_torch_cuda"])
    if loader_gen is not None and st.get("rng_loader_generator") is not None:
        loader_gen.set_state(st["rng_loader_generator"])
    if pair_gen is not None and st.get("rng_pair_generator") is not None:
        pair_gen.set_state(st["rng_pair_generator"])


def rng_fingerprint(st: dict[str, Any]) -> dict[str, str]:
    """Short hashes to verify that evaluation left training RNG untouched."""
    import hashlib  # noqa: PLC0415

    def h(x: Any) -> str:
        if x is None:
            return "none"
        if isinstance(x, torch.Tensor):
            return hashlib.sha256(x.numpy().tobytes()).hexdigest()[:16]
        if isinstance(x, list):
            return hashlib.sha256(b"".join(t.numpy().tobytes() for t in x)).hexdigest()[:16]
        return hashlib.sha256(repr(x).encode()).hexdigest()[:16]

    return {k: h(v) for k, v in st.items()}


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    # RNG state objects (numpy tuples, python tuples) require weights_only=False; the file is our own artifact.
    return torch.load(str(path), map_location=map_location, weights_only=False)
