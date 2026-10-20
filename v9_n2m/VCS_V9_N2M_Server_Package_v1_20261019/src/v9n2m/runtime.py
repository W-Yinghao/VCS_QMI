"""Stage a verified copy of the inherited host; never edit the old source tree."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from .common import PACKAGE_ROOT, assert_same_file, canonical_sha256, read_json, sha256_file, utc_now, write_json
from .identity import pinned_freeze, verify_source

ENVIRONMENT_KEYS = (
    "gpu", "torch", "cuda", "python", "numpy", "cudnn_version", "driver",
    "deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark",
    "tf32", "cudnn_tf32", "CUBLAS_WORKSPACE_CONFIG", "num_threads",
    "float32_matmul_precision", "torch_build_sha256",
)


def package_code_identity():
    paths = list((PACKAGE_ROOT / "src").rglob("*.py"))
    paths += [PACKAGE_ROOT / "run_n2m.py", PACKAGE_ROOT / "config/protocol.json"]
    paths += list((PACKAGE_ROOT / "scripts").glob("*.py"))
    paths += list((PACKAGE_ROOT / "scripts").glob("*.sbatch"))
    paths += list((PACKAGE_ROOT / "provenance").glob("*.json"))
    mapping = {str(p.relative_to(PACKAGE_ROOT)): sha256_file(p) for p in sorted(paths)}
    return {"sha256": mapping, "digest": canonical_sha256(mapping)}


def bootstrap_runtime(base, out):
    from .trainer_patch import patch_train_source
    base, out = Path(base).resolve(), Path(out).resolve()
    if out == base or base in out.parents:
        raise ValueError("New output must be outside the immutable original handoff")
    frozen = pinned_freeze(base)
    verify_source(base, include_cache=False)
    runtime = out / "runtime"
    receipt = runtime / "RUNTIME.json"
    if receipt.exists():
        info = verify_runtime_files(runtime)
        if info["base_freeze_sha256"] != sha256_file(base / "FROZEN.json"):
            raise ValueError("Existing runtime belongs to another N2F freeze")
        expected_patch = patch_train_source((base / "source/src/v9/n2/train.py").read_text())
        if (runtime / "src/v9/n2/train.py").read_text() != expected_patch:
            raise ValueError("Existing runtime uses a different telemetry patch")
        return info
    if runtime.exists() and any(runtime.iterdir()):
        raise ValueError("Unreceipted partial runtime exists; archive it before restaging")
    runtime.mkdir(parents=True, exist_ok=True)
    inherited, effective = {}, {}
    for rel, digest in frozen["code_sha256"].items():
        if not rel.startswith("source/") or rel.startswith("source/token_cache/"):
            continue
        # REG and utility source may be copied as provenance but no REG entrypoint is exposed.
        suffix = rel[len("source/"):]
        dest = runtime / suffix
        dest.parent.mkdir(parents=True, exist_ok=True)
        assert_same_file(base / rel, digest)
        if suffix == "src/v9/n2/train.py":
            dest.write_text(patch_train_source((base / rel).read_text()), encoding="utf-8")
        else:
            shutil.copyfile(base / rel, dest)
        inherited[suffix] = digest
        effective[suffix] = sha256_file(dest)
    info = {"schema_version": 1, "created_utc": utc_now(), "base": str(base), "runtime": str(runtime),
            "base_freeze_sha256": sha256_file(base / "FROZEN.json"), "source_git": frozen["source_git"],
            "inherited_sha256": inherited, "effective_sha256": effective,
            "effective_digest": canonical_sha256(effective),
            "patch_module_sha256": sha256_file(PACKAGE_ROOT / "src/v9n2m/trainer_patch.py"),
            "changed_source_files": [p for p in effective if effective[p] != inherited[p]],
            "cache_note": "Training reads verified original frozen token cache; no cache generation is needed."}
    if info["changed_source_files"] != ["src/v9/n2/train.py"]:
        raise ValueError("Unexpected scientific source mutation during staging")
    write_json(receipt, info)
    return info


def verify_runtime_files(runtime):
    runtime = Path(runtime).resolve()
    info = read_json(runtime / "RUNTIME.json")
    importable = {str(p.relative_to(runtime)) for p in runtime.rglob("*")
                  if p.is_file() and (p.suffix in (".py", ".so", ".pyd")
                                      or (p.suffix == ".pyc" and "__pycache__" not in p.parts))}
    expected_importable = {rel for rel in info["effective_sha256"] if Path(rel).suffix in (".py", ".so", ".pyd")}
    if importable != expected_importable:
        raise ValueError(f"Unregistered or missing importable runtime files: {sorted(importable ^ expected_importable)}")
    for rel, expected in info["effective_sha256"].items():
        p = runtime / rel
        if runtime not in p.resolve().parents:
            raise ValueError("Runtime identity contains an escaping path")
        assert_same_file(p, expected)
    if canonical_sha256(info["effective_sha256"]) != info["effective_digest"]:
        raise ValueError("Runtime file-map digest changed")
    return info


def configure_precision():
    # Set before importing Torch / initializing CUDA.
    expected = ":4096:8"
    old = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if old not in (None, expected):
        raise ValueError(f"Unexpected CUBLAS_WORKSPACE_CONFIG={old!r}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = expected
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def collect_environment():
    import numpy as np
    import torch
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10, check=True)
        versions = sorted(set(proc.stdout.strip().splitlines()))
        driver = versions[0] if len(versions) == 1 else versions
    except (OSError, subprocess.SubprocessError):
        driver = None
    return {"host": platform.node(), "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "python": platform.python_version(), "numpy": np.__version__,
            "cudnn_version": torch.backends.cudnn.version(), "driver": driver,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "num_threads": torch.get_num_threads(), "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "torch_build_sha256": canonical_sha256(torch.__config__.show()),
            "job_id": os.environ.get("SLURM_JOB_ID"), "observed_utc": utc_now()}


def environment_differences(expected, actual):
    differences = {}
    for k in ENVIRONMENT_KEYS:
        if k not in expected or expected[k] is None:
            differences[k] = {"expected": expected.get(k), "actual": actual.get(k), "reason": "historical_or_target_value_not_recorded"}
        elif k not in actual or actual[k] is None or expected[k] != actual[k]:
            differences[k] = {"expected": expected[k], "actual": actual.get(k), "reason": "different_or_missing"}
    return differences
