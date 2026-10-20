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
            "CoMIR_native", "RuLSIF_exact", "MIND_2D_adapted", "RuLSIF_exact_v2")
# RuLSIF_exact_v2 is a separately registered variant designed after v1's CAL failure. Its
# grid is read from the committed pre-registration record (part of the source inventory).
RULSIF_V2_RECORD = REPO / "v10/supplement/RULSIF_V2_PREREGISTRATION.json"
RULSIF_V2_FROZEN = ("FROZEN_BEFORE_CAL",)
# Labelled post-gate amendment (separate record; the pre-registration record stays pinned).
RULSIF_V2_AMENDMENT = REPO / "v10/supplement/RULSIF_V2_AMENDMENT_20261019.json"
RULSIF_V2_AMENDMENT_STATUS = "OWNER_DELEGATED_POST_GATE_AMENDMENT"
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
    elif family == "RuLSIF_exact_v2":
        return rulsif_v2_eligible()
    else:
        raise ValueError("Unknown family: " + family)
    return [{"order": i, **row} for i, row in enumerate(rows)]


def rulsif_v2_grid(record=None):
    """The frozen v2 candidate grid, exactly as pre-registered (orders 0..n-1, at most 9)."""
    record = read(RULSIF_V2_RECORD) if record is None else record
    if record.get("family") != "RuLSIF_exact_v2" or record.get("status") not in RULSIF_V2_FROZEN:
        raise RuntimeError("RuLSIF_exact_v2 candidate grid is not frozen in its pre-registration record")
    grid = record.get("candidate_grid")
    if not isinstance(grid, list) or not 1 <= len(grid) <= 9:
        raise RuntimeError("RuLSIF_exact_v2 grid must hold 1-9 pre-registered candidates")
    if [row.get("order") for row in grid] != list(range(len(grid))):
        raise RuntimeError("RuLSIF_exact_v2 candidate order must be 0..n-1")
    if record.get("candidate_grid_sha256") != digest(grid):
        raise RuntimeError("RuLSIF_exact_v2 grid differs from its recorded digest")
    definition = record.get("definition", {})
    for row in grid:
        if (row.get("alpha") != 0.5 or row.get("constant_basis") is not True
                or row.get("fit_support") != "disk_r129" or row.get("patch_radius") != 4):
            raise RuntimeError("RuLSIF_exact_v2 candidate leaves the registered definition")
        if any(row.get(key) != value for key, value in definition.items()):
            raise RuntimeError("RuLSIF_exact_v2 candidate differs from the record's fixed definition")
    return [dict(row) for row in grid]


def rulsif_v2_amendment(grid=None, amendment=None, record_sha256=None):
    """The validated post-gate amendment, or None when no amendment record exists."""
    if amendment is None:
        if not RULSIF_V2_AMENDMENT.exists():
            return None
        amendment = read(RULSIF_V2_AMENDMENT)
    grid = rulsif_v2_grid() if grid is None else grid
    record_sha256 = sha(RULSIF_V2_RECORD) if record_sha256 is None else record_sha256
    if amendment.get("family") != "RuLSIF_exact_v2" or amendment.get("status") != RULSIF_V2_AMENDMENT_STATUS:
        raise RuntimeError("Not a RuLSIF_exact_v2 post-gate amendment")
    if amendment.get("amends", {}).get("preregistration_record", {}).get("sha256") != record_sha256:
        raise RuntimeError("Amendment names a different pre-registration record")
    rule = amendment.get("amended_rule", {})
    orders = rule.get("eligible_orders")
    if (not isinstance(orders, list) or not orders or orders != sorted(set(orders))
            or not set(orders) <= {row["order"] for row in grid}):
        raise RuntimeError("Eligible orders must be a nonempty increasing subset of the frozen grid")
    if not set(rule.get("required_levels", [])) <= {"deployed_position_heldout", "crossfit4_core_heldout"} \
            or not rule.get("required_levels"):
        raise RuntimeError("Unknown or empty required gate levels")
    by_order = {row["order"]: row for row in grid}
    eligible = amendment.get("eligible_candidates")
    if eligible != [by_order[o] for o in orders] or amendment.get("eligible_candidates_sha256") != digest(eligible):
        raise RuntimeError("Eligible candidates differ from the frozen grid rows or their digest")
    return amendment


def rulsif_v2_eligible(grid=None, amendment=None, record_sha256=None):
    """CAL candidates: the frozen grid, or its amended eligible subset in the frozen order."""
    grid = rulsif_v2_grid() if grid is None else grid
    amendment = rulsif_v2_amendment(grid, amendment, record_sha256)
    if amendment is None:
        return grid
    return [dict(row) for row in amendment["eligible_candidates"]]


def schedule(family):
    if family in HOST_METHODS:
        return {"kind": "gradient_training", "steps": 3000,
                "save_steps": list(range(250, 3001, 250)), "cal_steps": list(range(250, 3001, 250)),
                "final_seeds": [1001, 1002, 1003]}
    if family == "CoMIR_native":
        return {"kind": "native_gradient_training", "steps": 23 * 32,
                "save_steps": list(range(32, 23 * 32 + 1, 32)), "cal_steps": [23 * 32],
                "final_seeds": [1001, 1002, 1003]}
    if family in ("RuLSIF_exact", "RuLSIF_exact_v2"):
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
        if (policy.get("policy_id") not in {"v10_ready_dispatch_20261019", "v10_method_pipeline_20261019"}
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


RULSIF_V2_RELEASE_STATUS = "FROZEN_PREREGISTERED_GATED"


def require_release_v2(path, family="RuLSIF_exact_v2"):
    """Minimal explicit release for the one family RuLSIF_exact_v2 (documented in README_CN.md).

    Differs from require_release only where v2's pre-registration differs: acceptance is the
    FIT-only functional gate (CPU) plus the CUDA T01/execution gate for this exact code; the
    end-to-end T02/runner acceptance is carried by the first formal CAL candidate itself
    (owner review 2026-10-19: no separate full-length validation round), so no seed-1117
    full CAL is required. Code identity, parent protocol, recipe, data lock, the active
    scheduling-policy pointer and the V10 pause marker are checked exactly as for v1.
    """
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Scientific execution requires Slurm")
    return verify_release_v2(path, os.environ.get("V10_SUPPLEMENT_EXPECT_RELEASE"), family)


def verify_release_v2(path, expected, family="RuLSIF_exact_v2"):
    """All release checks except the Slurm requirement (also used by the release builder)."""
    if family != "RuLSIF_exact_v2":
        raise RuntimeError("The v2 release applies to RuLSIF_exact_v2 only")
    path = Path(path).resolve()
    if not expected or sha(path) != expected:
        raise RuntimeError("Missing or changed submission-pinned execution release")
    release = read(path)
    if release.get("status") != RULSIF_V2_RELEASE_STATUS or release.get("family") != family:
        raise RuntimeError("Not a gated RuLSIF_exact_v2 release")
    inventory = source_inventory()
    code = digest(inventory)
    if release.get("source_files") != inventory or release.get("code_sha256") != code:
        raise RuntimeError("Supplemental source identity changed")
    if sha(PARENT / "PROTOCOL.json") != PARENT_PROTOCOL_SHA256 or release.get("parent_protocol_sha256") != PARENT_PROTOCOL_SHA256:
        raise RuntimeError("Parent common protocol changed")
    if release.get("families", {}).get(family) != {"candidates": candidates(family), "schedule": schedule(family)}:
        raise RuntimeError("Method attachment differs from the pre-registered recipe")
    record_sha = sha(RULSIF_V2_RECORD)
    if release.get("preregistration", {}).get("sha256") != record_sha:
        raise RuntimeError("Pre-registration record differs from the release")
    acceptance = release.get("acceptance", {})
    for stage, report_stage in (("FIT_GATE", "FIT_FUNCTIONAL_GATE"), ("GPU_GATE", "T01_AND_EXECUTION_GATE")):
        ref = acceptance.get(stage) or {}
        if not ref.get("path") or sha(ref["path"]) != ref.get("sha256"):
            raise RuntimeError("Acceptance record changed: " + stage)
        report = read(ref["path"])
        if (report.get("status") != "PASS" or report.get("stage") != report_stage
                or report.get("family") != family or report.get("code_sha256") != code
                or report.get("held_test_read") is not False or report.get("dev_read") is not False):
            raise RuntimeError("Acceptance is not PASS for this exact code: " + stage)
        if stage == "FIT_GATE":
            check_fit_gate_v2(report, record_sha)
    amendment = rulsif_v2_amendment()
    if amendment is not None and release.get("amendment", {}).get("sha256") != sha(RULSIF_V2_AMENDMENT):
        raise RuntimeError("Release does not pin the post-gate amendment")
    policy_ref = release.get("scheduling_authorization") or {}
    policy_path = Path(policy_ref.get("path", ""))
    if not policy_ref.get("path") or sha(policy_path) != policy_ref.get("sha256"):
        raise RuntimeError("Scheduling authorization changed")
    policy = read(policy_path)
    active = read(PROJECT / "handoff/ACTIVE_SCHEDULING_POLICY.json")
    if (active.get("policy_id") != policy.get("policy_id")
            or Path(active.get("policy", "")).resolve() != policy_path.resolve()
            or policy.get("validated_supplement_overlap") is not True
            or policy.get("normal_gpu_limit") != 8):
        raise RuntimeError("Active policy does not authorize validated V10 overlap")
    if (PROJECT / "handoff/v10_parallel8_20261018/PAUSE_SUBMISSIONS").exists():
        raise RuntimeError("The current scheduling policy still pauses new V10 work")
    ref = release.get("data_lock") or {}
    if not ref.get("path") or sha(ref["path"]) != ref.get("sha256"):
        raise RuntimeError("Pinned input changed: data_lock")
    return release


def check_fit_gate_v2(report, record_sha, family="RuLSIF_exact_v2"):
    """FIT gate coverage: every frozen candidate evaluated; under the amendment, exactly the declared
    eligible candidates pass the required levels and the re-run reproduces the original verdicts."""
    entries = report.get("families", {}).get(family, {})
    if report.get("preregistration_sha256") != record_sha or entries.get("status") != "PASS":
        raise RuntimeError("FIT functional gate is not PASS for this pre-registration record")
    evaluated = [e.get("candidate") for e in entries.get("candidates", [])]
    if evaluated != rulsif_v2_grid():
        raise RuntimeError("FIT functional gate did not evaluate the whole frozen grid")
    amendment = rulsif_v2_amendment()
    passing = [e["candidate"] for e in entries["candidates"] if e.get("status") == "PASS"]
    if amendment is None:
        if passing != evaluated or report.get("rule_mode") != "preregistered":
            raise RuntimeError("FIT functional gate does not pass every pre-registered candidate")
        return
    rule = amendment["amended_rule"]
    reproduction = report.get("reproduction") or {}
    if (report.get("rule_mode") != "amended" or report.get("amendment_sha256") != sha(RULSIF_V2_AMENDMENT)
            or report.get("required_levels") != rule["required_levels"]
            or passing != candidates(family) or entries.get("eligible_orders") != rule["eligible_orders"]
            or reproduction.get("verdict_differences") != []
            or reproduction.get("original_report_sha256")
            != amendment["amends"]["original_gate_reports"][family]["sha256"]):
        raise RuntimeError("FIT functional gate does not match the post-gate amendment")