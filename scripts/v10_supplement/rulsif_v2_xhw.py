"""Cross-hardware final seeds for RuLSIF_exact_v2 (owner decision 2026-10-19: 「final 放开，CAL 留 L40S」).

One release-version delta from the CAL release 41313c39 (source 2e6ea58): the final seeds accept the
GPU models NVIDIA L40S, NVIDIA RTX PRO 6000 Blackwell Server Edition and NVIDIA H100 NVL, named as the
CUDA device reports them. CAL stays L40S-only and cannot run under this release. Every other
environment and precision check is unchanged. The CAL selection made under 41313c39 is consumed
after its hash is verified; it is never re-selected. A seed whose final belongs to 41313c39 stays
valid and is refused here. A final on RTX PRO 6000 or H100 needs a PASS T01/execution gate on that
model for this exact code (validated under the pinned cross-hardware manifest); L40S uses the CAL
release's gate, whose science code is byte-identical. Each final runs on one GPU model and records
the model, node and partition. Standard library only.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v10.supplement.contract import (PARENT, PARENT_PROTOCOL_SHA256, PROJECT, RULSIF_V2_AMENDMENT,  # noqa: E402
                                     RULSIF_V2_RECORD, candidates, digest, read, schedule, sha,
                                     source_inventory)
from v10.supplement.records import write  # noqa: E402

FAMILY = "RuLSIF_exact_v2"
CAL_GPU_NAMES = ("NVIDIA L40S",)
FINAL_GPU_NAMES = ("NVIDIA L40S", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA H100 NVL")
PARTITION_OF = {"NVIDIA L40S": "L40S", "NVIDIA RTX PRO 6000 Blackwell Server Edition": "RTX6000PRO",
                "NVIDIA H100 NVL": "H100"}
XHW_RELEASE_STATUS = "FROZEN_PREREGISTERED_GATED_XHW_FINALS"
# Facts of the CAL release this version extends (all re-verified against the files at run time).
BASE_COMMIT = "2e6ea58286f938bd04e68451e4b42d635d4cc12e"
BASE_CODE_SHA256 = "f6ea2af626a2c233062993e77242a8201897c06bf2c0ccbb5d43e592d70055e5"
BASE_RELEASE_SHA256 = "41313c395898f99d911b1e92532cd5fe809f1563165fdbfefa8673859bd6e9be"
SELECTION_SHA256 = "cea56cbc1e676e293588e548158e7a4897a4ee253c5a9a7b3c614cbd6e68b919"
CAL_JOB = "1049724"
BUNDLE = PROJECT / "handoff/v10_rulsif_v2_20261019"
XHW_DIR = BUNDLE / "xhw"
# The only files allowed to differ from the CAL release's source inventory.
CHANGED = ("scripts/v10_supplement/run.py", "scripts/v10_supplement/validate.py")
ADDED = ("scripts/v10_supplement/rulsif_v2_xhw.py", "scripts/v10_supplement/rulsif_v2_xhw_release.py",
         "tests/v10/test_rulsif_v2_xhw.py")
OWNER_DECISION = {
    "chosen_option_verbatim": "final 放开，CAL 留 L40S",
    "offered_as_verbatim": "EPFL final、BF 和 EPFL 的 RuLSIF v2 final、K8/K32 剩下的 6 个单元，改为三分区，记录实际硬件、分层报告。"
                           "CAL 和计时留在 L40S，保持每族 8 个 L40S GPU 小时的同等预算。先跑跨硬件门禁，再撤下未开始的作业重新提交。",
    "standing_instruction_0931_utc_verbatim": "后续待运行GPU任务统一 --partition=RTX6000PRO,L40S,H100，不要再按seed、方法或GPU型号"
                                              "人为分组/限卡；记录实际硬件与成本即可"}
REPORTING_RULE = ("Hardware is recorded per final (GPU model, node, partition, allocated time). Final results are "
                  "reported per hardware stratum and pooled; no claim of cross-hardware equivalence is made. CAL "
                  "selection and its budget stay on L40S (8 L40S GPU-h per family).")


def delta(base_files, current_files):
    """The source delta against the CAL release: only the hardware plumbing may change."""
    changed = sorted(p for p in base_files if p in current_files and base_files[p] != current_files[p])
    added = sorted(set(current_files) - set(base_files))
    removed = sorted(set(base_files) - set(current_files))
    science = [p for p in changed + added + removed
               if p.startswith(("src/", "v10/supplement/")) or (p.startswith("tests/") and p not in ADDED)]
    return {"changed": changed, "added": added, "removed": removed,
            "unchanged": len([p for p in base_files if current_files.get(p) == base_files[p]]),
            "allowlisted_only": set(changed) <= set(CHANGED) and set(added) <= set(ADDED) and not removed,
            "science_identical": not science, "science_files_touched": science}


def is_xhw_release(path):
    try:
        return read(path).get("status") == XHW_RELEASE_STATUS
    except (OSError, ValueError):
        return False


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def require_xhw_release(path):
    require(bool(os.environ.get("SLURM_JOB_ID")), "Scientific execution requires Slurm")
    return verify_xhw_release(path, os.environ.get("V10_SUPPLEMENT_EXPECT_RELEASE"))


def verify_xhw_release(path, expected):
    """Every check of the cross-hardware final release except the Slurm requirement."""
    path = Path(path).resolve()
    require(bool(expected) and sha(path) == expected, "Missing or changed submission-pinned execution release")
    release = read(path)
    require(release.get("status") == XHW_RELEASE_STATUS and release.get("family") == FAMILY,
            "Not a cross-hardware RuLSIF_exact_v2 final release")
    require(release.get("phases") == ["final"], "The cross-hardware release runs final seeds only")
    inventory = source_inventory()
    code = digest(inventory)
    require(release.get("source_files") == inventory and release.get("code_sha256") == code,
            "Supplemental source identity changed")
    base_path = BUNDLE / "RELEASE.json"
    require(sha(base_path) == BASE_RELEASE_SHA256
            and release.get("base_release") == {"path": str(base_path), "sha256": BASE_RELEASE_SHA256},
            "CAL release changed or not pinned")
    base = read(base_path)
    require(base.get("code_sha256") == BASE_CODE_SHA256 and base.get("source_commit") == BASE_COMMIT,
            "CAL release names other code")
    change = delta(base["source_files"], inventory)
    require(change["allowlisted_only"] and change["science_identical"],
            "Source delta exceeds the hardware amendment: " + repr(change["science_files_touched"] or change))
    review = release.get("delta_review") or {}
    require(bool(review.get("path")) and sha(review["path"]) == review.get("sha256"), "Delta review changed")
    reviewed = read(review["path"])
    require(reviewed.get("status") == "PASS" and reviewed.get("new_code_sha256") == code
            and reviewed.get("base_code_sha256") == BASE_CODE_SHA256, "Delta review is not PASS for this code")
    require(sha(PARENT / "PROTOCOL.json") == PARENT_PROTOCOL_SHA256
            and release.get("parent_protocol_sha256") == PARENT_PROTOCOL_SHA256 == base.get("parent_protocol_sha256"),
            "Parent common protocol changed")
    attachment = {"candidates": candidates(FAMILY), "schedule": schedule(FAMILY)}
    require(release.get("families", {}).get(FAMILY) == attachment == base["families"][FAMILY],
            "Method attachment differs from the CAL release")
    for key, current in (("preregistration", RULSIF_V2_RECORD), ("amendment", RULSIF_V2_AMENDMENT)):
        require(release.get(key, {}).get("sha256") == sha(current) == base[key]["sha256"],
                "Pre-registration or amendment differs from the CAL release: " + key)
    selection_path = BUNDLE / "results/CAL" / FAMILY / "SELECTION.json"
    require(sha(selection_path) == SELECTION_SHA256
            and release.get("cal_selection") == {"path": str(selection_path), "sha256": SELECTION_SHA256},
            "CAL selection changed or not pinned")
    selection = read(selection_path)
    require(selection.get("release_sha256") == BASE_RELEASE_SHA256 and selection.get("job") == CAL_JOB
            and selection.get("status") == "SELECTED", "CAL selection belongs to another release or job")
    receipt_path = BUNDLE / "exits" / f"{CAL_JOB}.json"
    receipt = read(receipt_path)
    require(receipt.get("job") == CAL_JOB and receipt.get("exit_code") == 0
            and receipt.get("release_sha256") == BASE_RELEASE_SHA256
            and release.get("cal_exit") == {"path": str(receipt_path), "sha256": sha(receipt_path)},
            "CAL exit receipt is not a pinned success under the CAL release")
    require(tuple(release.get("final_gpu_names", ())) == FINAL_GPU_NAMES
            and tuple(release.get("cal_gpu_names", ())) == CAL_GPU_NAMES, "Accepted GPU models changed")
    manifest = release.get("xhw_validation_manifest") or {}
    require(bool(manifest.get("path")) and sha(manifest["path"]) == manifest.get("sha256"),
            "Cross-hardware validation manifest changed")
    pinned = read(manifest["path"])
    require(tuple(pinned.get("accepted_gpu_names", ())) == FINAL_GPU_NAMES and pinned.get("code_sha256") == code
            and pinned.get("source_root") == str(REPO.resolve()), "Cross-hardware manifest names other code or models")
    base_gate = base["acceptance"]["GPU_GATE"]
    require(sha(base_gate["path"]) == base_gate["sha256"] and release.get("l40s_gate") == base_gate,
            "CAL release GPU gate changed")
    lock = release.get("data_lock") or {}
    require(lock == base.get("data_lock") and sha(lock["path"]) == lock.get("sha256"), "Pinned input changed: data_lock")
    policy_ref = release.get("scheduling_authorization") or {}
    policy_path = Path(policy_ref.get("path", ""))
    require(bool(policy_ref.get("path")) and sha(policy_path) == policy_ref.get("sha256"), "Scheduling authorization changed")
    policy = read(policy_path)
    active = read(PROJECT / "handoff/ACTIVE_SCHEDULING_POLICY.json")
    require(active.get("policy_id") == policy.get("policy_id")
            and Path(active.get("policy", "")).resolve() == policy_path.resolve()
            and policy.get("validated_supplement_overlap") is True and policy.get("normal_gpu_limit") == 8,
            "Active policy does not authorize validated V10 overlap")
    require(not (PROJECT / "handoff/v10_parallel8_20261018/PAUSE_SUBMISSIONS").exists(),
            "The current scheduling policy still pauses new V10 work")
    require(release.get("owner_decision") == OWNER_DECISION and release.get("reporting_rule") == REPORTING_RULE,
            "Owner decision or reporting rule differs")
    return release


def gate_for(gpu_name, release):
    """The PASS T01/execution gate that licenses a final on this GPU model (raises if none)."""
    require(gpu_name in release["final_gpu_names"], "GPU model outside the release: " + repr(gpu_name))
    if gpu_name == "NVIDIA L40S":
        ref = release["l40s_gate"]
        report = read(ref["path"])
        require(sha(ref["path"]) == ref["sha256"] and report.get("status") == "PASS"
                and report.get("environment", {}).get("gpu_name") == gpu_name
                and report.get("code_sha256") == BASE_CODE_SHA256, "CAL release L40S gate is not PASS")
        return {"gate_report": ref, "gate_job": report["job"], "gate_code_sha256": BASE_CODE_SHA256,
                "gate_basis": "CAL release gate; science code byte-identical (delta review)"}
    manifest = release["xhw_validation_manifest"]
    root = Path(manifest["path"]).parent
    for receipt_path in sorted((root / "exits").glob("*.json")):
        receipt = read(receipt_path)
        job = str(receipt.get("job"))
        if receipt.get("exit_code") != 0 or receipt.get("validation_manifest_sha256") != manifest["sha256"]:
            continue
        aggregate_path = root / "runs" / job / "GATE.json"
        if not aggregate_path.exists():
            continue
        aggregate = read(aggregate_path)
        if (aggregate.get("status") != "PASS" or aggregate.get("requested_families") != [FAMILY]
                or aggregate.get("code_sha256") != release["code_sha256"] or aggregate.get("job") != job
                or len(aggregate.get("families", [])) != 1):
            continue
        ref = aggregate["families"][0]
        if ref.get("family") != FAMILY or sha(ref["path"]) != ref["sha256"]:
            continue
        report = read(ref["path"])
        if (report.get("status") == "PASS" and report.get("stage") == "T01_AND_EXECUTION_GATE"
                and report.get("family") == FAMILY and report.get("code_sha256") == release["code_sha256"]
                and report.get("environment", {}).get("gpu_name") == gpu_name):
            return {"gate_report": {"path": ref["path"], "sha256": ref["sha256"]}, "gate_job": job,
                    "gate_receipt": {"path": str(receipt_path), "sha256": sha(receipt_path)},
                    "gate_code_sha256": release["code_sha256"], "gate_basis": "cross-hardware gate on this model"}
    raise RuntimeError(f"No PASS cross-hardware gate on {gpu_name} for this code; final refused on this model")


def prepare_final_dir(directory, ident, environment, gate):
    """Refuse a CAL-release or foreign seed directory; keep one GPU model per final; record hardware."""
    directory = Path(directory)
    existing = directory / "IDENTITY.json"
    if existing.exists():
        current = read(existing)
        require(current.get("release_sha256") != BASE_RELEASE_SHA256,
                "This seed's final belongs to the CAL release 41313c39; it stays valid and is not re-run")
        require(current == ident, "Seed directory belongs to another unit")
    hardware = {"gpu_name": environment["gpu_name"], "gpu_capability": environment.get("gpu_capability"),
                "node": environment.get("host"), "slurm_nodelist": environment.get("slurm_nodelist"),
                "partition": environment.get("slurm_partition"), **gate}
    path = directory / "HARDWARE.json"
    if path.exists():
        first = read(path)
        require(first.get("gpu_name") == hardware["gpu_name"],
                f"Final started on {first.get('gpu_name')}; a final runs on one GPU model (resubmit on that model)")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        write(path, {**hardware, "first_job": environment.get("slurm_job_id"),
                     "utc": time.strftime("%FT%TZ", time.gmtime()),
                     "rule": "one GPU model per final; resumptions must use the same model"})
    return hardware
