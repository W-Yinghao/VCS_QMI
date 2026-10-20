"""Provenance, freeze and record helpers for the post-DEV diagnostics (standard library only).

The frozen supplement trees are only ever read: every reused source file is checked
against the frozen RELEASE.json / validation MANIFEST before anything is imported.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
BUNDLE = Path(os.environ.get("V10D_BUNDLE", str(PROJECT / "handoff/v10_sensitivity_20261019")))
OUT_ROOT = Path(os.environ.get("V10D_OUT_ROOT",
                               "/projects/EEG-foundation-model/yinghao/outputs_v10/sensitivity_20261019"))
PREREGISTRATION = REPO / "v10/sensitivity/PREREGISTRATION.json"
LABEL = "post-DEV sensitivity (designed after the frozen DEV results were seen; never replaces a frozen row)"
FREEZE_PATHS = ("src/v10_sensitivity", "scripts/v10_sensitivity", "slurm/v10_sensitivity",
                "tests/v10_sensitivity", "v10/sensitivity")
DEV24_EXCLUDED = ("1B_F4", "1B_H4", "1B_H9", "2B_D5", "2B_G3", "2B_G5", "2B_H2", "2B_I5")
PARTITIONS = {"D1": ("L40S",), "D2": ("L40S",), "D3": ("RTX6000PRO", "H100")}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    """Atomic JSON write; refuses NaN/inf so a non-finite value can never be recorded silently."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_once(path, value):
    """Write a record that must never change: identical rewrite is a no-op, a different one refuses."""
    path = Path(path)
    if path.exists():
        require(read(path) == json.loads(json.dumps(value, allow_nan=False)),
                f"{path} already exists with different content")
        return
    write(path, value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def prereg(path=PREREGISTRATION):
    return read(path)


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=60)
    require(result.returncode == 0, f"git {' '.join(args)} failed in {root}: {result.stderr.strip()[:300]}")
    return result.stdout.strip()


def git_head(root):
    return git(root, "rev-parse", "HEAD")


def tracked_files(root=REPO, paths=FREEZE_PATHS):
    listed = git(root, "ls-files", "--", *paths)
    return sorted(line for line in listed.splitlines() if line)


def source_hashes(root=REPO, paths=FREEZE_PATHS):
    return {name: sha(Path(root) / name) for name in tracked_files(root, paths)}


def dirty(root=REPO, paths=FREEZE_PATHS):
    return git(root, "status", "--porcelain", "--untracked-files=all", "--", *paths)


def verify_freeze(expected_sha=None):
    """Job-time pin: FREEZE.json sha256 given at submission, then every listed diagnostics file."""
    expected_sha = expected_sha or os.environ.get("V10D_FREEZE_SHA256")
    require(bool(expected_sha), "V10D_FREEZE_SHA256 (submission pin of FREEZE.json) is required")
    path = BUNDLE / "FREEZE.json"
    require(path.exists() and sha(path) == expected_sha, "FREEZE.json missing or changed after submission")
    freeze = read(path)
    require(freeze.get("status") == "FROZEN_FOR_SUBMISSION", "FREEZE.json is not a submission freeze")
    for name, expected in freeze["diagnostic_sources"]["files"].items():
        require(sha(REPO / name) == expected, "Diagnostics source changed after freeze: " + name)
    require(sha(PREREGISTRATION) == freeze["preregistration"]["sha256"], "Pre-registration changed after freeze")
    require(sha(BUNDLE / "PROTOCOL.json") == freeze["preregistration"]["sha256"],
            "Bundle PROTOCOL.json is not the frozen pre-registration")
    return freeze


def verify_release(ref):
    """Frozen supplement release: pinned sha256, source root/commit/code identity, every source file, git HEAD."""
    path = Path(ref["path"])
    require(sha(path) == ref["sha256"], "Frozen release changed: " + str(path))
    release = read(path)
    root = Path(ref["source_root"])
    require(release["status"] == "FROZEN_VALIDATED", "Release is not FROZEN_VALIDATED")
    require(release["source_root"] == str(root), "Release names a different source tree")
    require(release["source_commit"] == ref["source_commit"], "Release names a different commit")
    require(release["code_sha256"] == ref["code_sha256"], "Release names a different code identity")
    require(release["data_lock"]["sha256"] == "e65c60a33f095933e673613fc58f2754485933524259e548c6a0f5dc380f72c5",
            "Release names a different data lock")
    for name, expected in release["source_files"].items():
        require(sha(root / name) == expected, "Frozen source file changed: " + name)
    require(git_head(root) == ref["source_commit"], "Frozen tree HEAD moved")
    return release


def verify_manifest(ref):
    """Frozen validation manifest (D3): pinned sha256, source files of its tree, git HEAD."""
    path = Path(ref["path"])
    require(sha(path) == ref["sha256"], "Validation manifest changed: " + str(path))
    manifest = read(path)
    root = Path(ref["source_root"])
    require(manifest["status"] == "VALIDATION_ONLY_NOT_A_SCIENTIFIC_RELEASE", "Not a validation-only manifest")
    require(manifest["source_root"] == str(root), "Manifest names a different source tree")
    for name, expected in manifest["source_files"].items():
        require(sha(root / name) == expected, "Frozen source file changed: " + name)
    require(git_head(root) == manifest["source_commit"], "Frozen tree HEAD moved")
    return manifest


def require_partition(diagnostic, partition=None):
    partition = partition if partition is not None else os.environ.get("SLURM_JOB_PARTITION")
    require(partition in PARTITIONS[diagnostic],
            f"{diagnostic} is pre-registered for partitions {PARTITIONS[diagnostic]}, got {partition!r}")
    return partition


def tma_position(core):
    return core.split("_", 1)[1]


def dev24_rows(records, fit_cores, cal_cores):
    """The frozen runner's DEV24 rule (scripts/v10_supplement/run.py), cross-checked against decision 6.1."""
    excluded = {tma_position(c) for c in list(fit_cores) + list(cal_cores)}
    kept = [r for r in records if tma_position(r["core"]) not in excluded]
    dropped = sorted({r["core"] for r in records} - {r["core"] for r in kept})
    require(len({r["core"] for r in kept}) == 24, "Frozen DEV24 sensitivity set changed")
    require(tuple(dropped) == DEV24_EXCLUDED, "DEV24 exclusions differ from decision 6.1")
    return kept
