"""Shared entry-point checks for the EPFL GPU jobs (item-5 gate and the full-cost timing): Slurm, commit pin, activation-record pin, the
common Torch / CUDA / cuDNN / precision environment of the accepted V10 runs (decision 3, the BF1 PROTOCOL.json precision block), the
actual GPU.  Standard library + torch."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from .guard import ACTIVATION_RECORD, EPFLDataGuard

REPO = Path(__file__).resolve().parents[3]
PARENT_PROTOCOL = Path("/home/infres/yinwang/CS_QMI/handoff/v10_bf1_20261017/PROTOCOL.json")
PARENT_PROTOCOL_SHA256 = "de4c0c3943d5a54b3a7de73a84785f240f7ee5b1396d173fd2109999a5555e9d"
ACCEPTED_GPUS = {"NVIDIA L40S": "L40S", "NVIDIA RTX PRO 6000 Blackwell Server Edition": "RTX6000PRO", "NVIDIA H100 NVL": "H100"}
BUDGET_GPU = "NVIDIA L40S"          # decision 8.2: family caps are L40S allocated GPU hours
OUT_ROOT = Path(os.environ.get("V10_EPFL_OUT", "/projects/EEG-foundation-model/yinghao/outputs_v10/epfl_activation"))
DATA_LOCK = REPO / "v10" / "epfl" / "EPFL_DATA_LOCK.json"
VALIDATION_SEED = 1117              # implementation checks / timing only (as the BF supplement validation); never 1017 or 1001-1003


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def job_pins() -> dict:
    """Slurm + the commit and activation-record pins exported at submission (slurm/v10_epfl/*.sbatch)."""
    require(bool(os.environ.get("SLURM_JOB_ID")), "EPFL GPU jobs run only through Slurm")
    commit = git("rev-parse", "HEAD")
    require(commit == os.environ.get("V10_EPFL_EXPECT_COMMIT"), f"HEAD {commit} != submitted {os.environ.get('V10_EPFL_EXPECT_COMMIT')}")
    require(git("status", "--porcelain", "--", "src/v10", "scripts/v10_epfl", "slurm/v10_epfl", "src/v9/reg", "v10/epfl") == "",
            "uncommitted changes under the EPFL code / records")
    record = Path(os.environ.get("V10_EPFL_RECORD", str(ACTIVATION_RECORD)))
    pin = os.environ.get("V10_EPFL_RECORD_SHA256")
    require(bool(pin) and sha256_file(record) == pin, "activation record differs from the submission pin")
    return {"commit": commit, "record": str(record), "record_sha256": pin, "job": os.environ["SLURM_JOB_ID"]}


def guard_from_pins(pins) -> EPFLDataGuard:
    lock = json.loads(DATA_LOCK.read_text())
    rec = json.loads(Path(pins["record"]).read_text())
    require(sha256_file(DATA_LOCK) == rec.get("data_lock_sha256"), "data lock differs from the activation record")
    return EPFLDataGuard(record=pins["record"], expect_record_sha256=pins["record_sha256"], data_lock=lock)


def runtime_environment(A) -> dict:
    """A.environment('cuda') checked against the accepted common environment; the GPU must be one of the cluster's three accepted types and
    is recorded (costs are reported per actual GPU; only L40S rows are in the decision-8.2 budget unit)."""
    require(sha256_file(PARENT_PROTOCOL) == PARENT_PROTOCOL_SHA256, "parent common protocol changed")
    expected = json.loads(PARENT_PROTOCOL.read_text())["precision"]
    env = A.environment("cuda")
    require(env.get("gpu_name") in ACCEPTED_GPUS, f"unexpected GPU {env.get('gpu_name')}")
    for key in ("torch", "cuda_runtime", "cudnn", "torch_threads"):
        require(env[key] == expected["environment"][key], f"environment mismatch: {key} {env[key]} != {expected['environment'][key]}")
    require(A.precision_settings(env["precision"]) == expected["settings"], "precision settings differ from the accepted environment")
    require(env["precision"]["env"] == expected["env"], "precision environment variables differ")
    env["budget_unit_gpu"] = env["gpu_name"] == BUDGET_GPU
    env["partition_label"] = ACCEPTED_GPUS[env["gpu_name"]]
    return env


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str, allow_nan=False) + "\n"); os.replace(tmp, path)
