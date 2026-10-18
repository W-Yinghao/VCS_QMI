"""Atomic provenance, exclusive writers, and complete-result identity checks."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import time

from .contract import read, sha, schedule


def utc():
    return time.strftime("%FT%TZ", time.gmtime())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def metrics(records):
    errors = [row["err_full"] for row in records]
    if not errors or not all(math.isfinite(error) and error >= 0 for error in errors):
        raise ValueError("Incomplete or invalid registration errors")
    return {"n": len(errors), "success42": sum(e < 42 for e in errors) / len(errors),
            "success9": sum(e < 9 for e in errors) / len(errors), "median_err": statistics.median(errors)}


def selection_key(row):
    m = row["metrics"]
    return -m["success42"], -m["success9"], m["median_err"], row["step"], row.get("order", 0)


def require_identity(directory, identity):
    path = Path(directory) / "IDENTITY.json"
    if path.exists() and read(path) != identity:
        raise RuntimeError("Output namespace already belongs to a different unit")
    if not path.exists():
        write(path, identity)


def verified_complete(directory, identity):
    directory = Path(directory)
    path = directory / "COMPLETE.json"
    if not path.exists():
        return None
    result = read(path)
    if result.get("status") != "COMPLETE" or result.get("identity") != identity:
        raise RuntimeError("Stale COMPLETE identity")
    recipe = schedule(identity["family"])
    if result.get("steps") != recipe["steps"]:
        raise RuntimeError("Incomplete training marked COMPLETE")
    if [r["step"] for r in result.get("cal_checkpoints", [])] != recipe["cal_steps"]:
        raise RuntimeError("Missing prescribed CAL checkpoints")
    if result.get("selected") != min(result["cal_checkpoints"], key=selection_key):
        raise RuntimeError("Checkpoint selection differs from complete CAL readouts")
    artifacts = result.get("artifacts", {})
    required = {"IDENTITY.json", "CONFIG.json", "DATA_READS.json"}
    for step in recipe["save_steps"]:
        required.update({f"checkpoints/step_{step:04d}.pt", f"checkpoints/step_{step:04d}.json"})
    for step in recipe["cal_steps"]:
        required.update({f"CAL/step_{step:04d}.jsonl", f"CAL/step_{step:04d}.summary.json"})
    if identity["phase"] == "FINAL_DEV":
        required.update({"DEV.jsonl", "DEV.summary.json"})
    if not required <= artifacts.keys():
        raise RuntimeError("COMPLETE lacks mandatory artifact entries")
    for name, expected in artifacts.items():
        artifact = (directory / name).resolve()
        if not artifact.is_relative_to(directory.resolve()) or sha(artifact) != expected:
            raise RuntimeError("COMPLETE artifact mismatch: " + name)
    return result


class BudgetExceeded(RuntimeError):
    pass


class Allocation:
    def __init__(self, root, family, phase):
        self.path = Path(root) / "COST_LEDGER.jsonl"
        self.family, self.phase = family, phase
        self.job = os.environ["SLURM_JOB_ID"]
        self.started = float(os.environ["V10_SUPPLEMENT_JOB_START"])
        previous = {}
        for row in rows(self.path):
            if row["family"] == family and row["phase"] == "CAL" and row["job"] != self.job:
                previous[row["job"]] = max(previous.get(row["job"], 0), row["allocated_seconds"])
        if phase == "CAL":
            for job in previous:
                receipt_path = Path(root) / "exits" / f"{job}.json"
                if not receipt_path.exists():
                    raise RuntimeError("Prior CAL allocation lacks an exit receipt; reconcile actual Slurm exit before retry")
                receipt = read(receipt_path)
                if (receipt.get("job") != job or receipt.get("release_sha256") !=
                        os.environ.get("V10_SUPPLEMENT_EXPECT_RELEASE")):
                    raise RuntimeError("Prior CAL exit receipt identity differs")
                seconds = receipt.get("allocated_seconds")
                if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                    raise RuntimeError("Invalid final allocation duration")
                previous[job] = max(previous[job], seconds)
        self.previous = sum(previous.values()) if phase == "CAL" else 0
        self.heartbeat = time.monotonic()
        self.record("START")

    def elapsed(self):
        return max(0, time.time() - self.started)

    def record(self, event, **extra):
        append(self.path, {"utc": utc(), "job": self.job, "family": self.family, "phase": self.phase,
                           "event": event, "allocated_seconds": self.elapsed(), "gpu_count": 1,
                           "work_kind": schedule(self.family)["kind"],
                           "accounting": "batch-start wall; queue separate; abrupt-exit allocation needs reconciliation",
                           **extra})
        self.heartbeat = time.monotonic()

    def check(self):
        if time.monotonic() - self.heartbeat > 30:
            self.record("HEARTBEAT")
        if self.phase == "CAL" and self.previous + self.elapsed() >= 8 * 3600 - 30:
            raise BudgetExceeded("The 8 L40S allocated GPUh CAL family cap is exhausted")
