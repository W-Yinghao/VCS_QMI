"""Method recipes and execution prerequisites; standard-library only.

These are implementation proposals, not scientific acceptance records. The
runner cannot execute them without separately pinned T01/T02/runner acceptance.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PROJECT = REPO.parent
PARENT = PROJECT / "handoff/v10_bf1_20261017"
PARENT_PROTOCOL_SHA256 = "de4c0c3943d5a54b3a7de73a84785f240f7ee5b1396d173fd2109999a5555e9d"
FAMILIES = ("MLP_VCS_perm8", "InfoNCE_host[J_perm1]", "MI_native", "NMI_native",
            "CoMIR_native", "RuLSIF_exact", "MIND_2D_adapted")
HOST_METHODS = {"MLP_VCS_perm8": "MLP_perm8", "InfoNCE_host[J_perm1]": "InfoNCE_host"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def candidates(family):
    if family in HOST_METHODS:
        rows = [{"lr": value} for value in (1e-4, 3e-4, 1e-3)]
    elif family in ("MI_native", "NMI_native"):
        rows = [{"bins": 64, "sigma_bins": 1.0, "range_percentiles": [0.5, 99.5]}]
    elif family == "CoMIR_native":
        rows = [{"epochs": 23, "steps_per_epoch": 32, "batch_size": 32, "lr": 0.01}]
    elif family == "RuLSIF_exact":
        rows = [{"bandwidth_multiplier": width, "ridge": ridge, "alpha": 0.5,
                 "centres": 128, "fit_positions_per_core": 2048, "patch_radius": 4}
                for width in (0.5, 1.0, 2.0) for ridge in (1e-4, 1e-3, 1e-2)]
    elif family == "MIND_2D_adapted":
        rows = [{"descriptor": {"radius": 1, "sigma": 0.8, "offset": 1, "variance_floor": 1e-6}}]
    else:
        raise ValueError("Unknown family: " + family)
    return [{"order": i, **row} for i, row in enumerate(rows)]


def schedule(family):
    if family in HOST_METHODS:
        return {"kind": "gradient_training", "steps": 3000,
                "save_steps": list(range(250, 3001, 250)), "cal_steps": list(range(250, 3001, 250)),
                "final_seeds": [1001, 1002, 1003]}
    if family == "CoMIR_native":
        return {"kind": "native_gradient_training", "steps": 23 * 32,
                "save_steps": list(range(32, 23 * 32 + 1, 32)), "cal_steps": [23 * 32],
                "final_seeds": [1001, 1002, 1003]}
    if family == "RuLSIF_exact":
        return {"kind": "kernel_fit", "steps": 1, "save_steps": [1], "cal_steps": [1],
                "final_seeds": [1001, 1002, 1003]}
    if family in ("MI_native", "NMI_native", "MIND_2D_adapted"):
        # No fake independent training seeds for deterministic training-free methods.
        return {"kind": "training_free", "steps": 0, "save_steps": [0], "cal_steps": [0],
                "final_seeds": [1001]}
    raise ValueError(family)


def source_inventory():
    paths = set()
    for relative in ("src/v10", "src/v9/reg", "scripts/v10_supplement", "tests/v10"):
        paths.update(p for p in (REPO / relative).rglob("*.py"))
    for suffix in ("*.sbatch", "*.sh"):
        paths.update((REPO / "slurm/v10_supplement").glob(suffix))
    paths.update((REPO / "v10/supplement").glob("*.json"))
    paths.add(REPO / "src/v9/__init__.py")
    return {str(p.relative_to(REPO)): sha(p) for p in sorted(paths)}


def require_release(path, family):
    """All paths are pinned before import/execution; no CLI override bypass."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Scientific execution requires Slurm")
    path = Path(path).resolve()
    expected = os.environ.get("V10_SUPPLEMENT_EXPECT_RELEASE")
    if not expected or sha(path) != expected:
        raise RuntimeError("Missing or changed submission-pinned execution release")
    release = read(path)
    if release.get("status") != "FROZEN_VALIDATED":
        raise RuntimeError("Supplemental implementation has not been validated/frozen")
    inventory = source_inventory()
    if release.get("source_files") != inventory or release.get("code_sha256") != digest(inventory):
        raise RuntimeError("Supplemental source identity changed")
    if sha(PARENT / "PROTOCOL.json") != PARENT_PROTOCOL_SHA256:
        raise RuntimeError("Parent common protocol changed")
    if release.get("parent_protocol_sha256") != PARENT_PROTOCOL_SHA256:
        raise RuntimeError("Execution release names a different common protocol")
    if release.get("families", {}).get(family) != {"candidates": candidates(family), "schedule": schedule(family)}:
        raise RuntimeError("Method attachment differs from implemented recipe")
    for stage in ("T01", "T02", "RUNNER"):
        ref = release["acceptance"][family][stage]
        report_path = path.parent / ref["path"]
        if sha(report_path) != ref["sha256"]:
            raise RuntimeError("Acceptance record changed: " + stage)
        report = read(report_path)
        if (report.get("status") != "PASS" or report.get("code_sha256") != digest(inventory)
                or report.get("family") != family):
            raise RuntimeError("Acceptance is not PASS for this exact code: " + stage)
    # The user's 2026-10-19 instruction releases ready V10 methods during the
    # V9 tail. This changes scheduling only; method acceptance above remains
    # mandatory. Keep the original strict path for releases without this
    # explicit, pinned scheduling authorization.
    authorization = release.get("scheduling_authorization")
    if authorization is None:
        audit_ref = release["v9_closeout"]
        audit_path = Path(audit_ref["path"])
        if sha(audit_path) != audit_ref["sha256"]:
            raise RuntimeError("V9 closeout changed")
        audit = read(audit_path)
        if audit.get("status") != "PASS" or len(audit.get("units", [])) != 33:
            raise RuntimeError("V9 full closeout is not complete")
    else:
        policy_path = Path(authorization["path"])
        if sha(policy_path) != authorization["sha256"]:
            raise RuntimeError("Scheduling authorization changed")
        policy = read(policy_path)
        active = read(PROJECT / "handoff/ACTIVE_SCHEDULING_POLICY.json")
        if (policy.get("policy_id") != "v10_ready_dispatch_20261019"
                or active.get("policy_id") != policy["policy_id"]
                or Path(active.get("policy", "")).resolve() != policy_path.resolve()
                or policy.get("validated_supplement_overlap") is not True
                or policy.get("v9_closeout_independent_obligation") is not True
                or policy.get("normal_gpu_limit") != 8):
            raise RuntimeError("Active policy does not authorize validated V10 overlap")
    if (PROJECT / "handoff/v10_parallel8_20261018/PAUSE_SUBMISSIONS").exists():
        raise RuntimeError("The current scheduling policy still pauses new V10 work")
    for key in ("data_lock", "comir_source"):
        ref = release[key]
        if sha(path.parent / ref["path"]) != ref["sha256"]:
            raise RuntimeError("Pinned input changed: " + key)
    return release
