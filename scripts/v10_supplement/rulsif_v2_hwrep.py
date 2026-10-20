"""Post-hoc hardware sensitivity diagnostic: RuLSIF_exact_v2 seed 1001 replicated on NVIDIA L40S.

Owner decisions 「final 放开，CAL 留 L40S」 and 「什么快什么来吧」. The primary seed-1001 final ran on NVIDIA RTX PRO
6000 (job 1050357) under RELEASE_XHW (12e37693). This diagnostic reruns exactly the same unit (identical
identity: RELEASE_XHW science, CAL selection cea56cbc, seed 1001) on L40S into a separate diagnostic root,
never the primary unit, and records per-registration agreement with the RTX run. It is not a fourth seed,
is not pooled with the primary results and never replaces the primary result. Standard library only;
also a CLI (delta-review, release, preflight, agreement) that submits nothing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/v10_supplement"))
import rulsif_v2_xhw as X  # noqa: E402
from v10.supplement.contract import digest, read, schedule, sha, source_inventory  # noqa: E402
from v10.supplement.records import metrics, rows, write  # noqa: E402

FAMILY, SEED = X.FAMILY, 1001
GPU_NAMES = ("NVIDIA L40S",)
STATUS = "POST_HOC_HARDWARE_SENSITIVITY_DIAGNOSTIC"
XHW_RELEASE = X.BUNDLE / "RELEASE_XHW.json"
XHW_RELEASE_SHA256 = "12e376931f4324c3f81096c87dc4bf7b22d90c75cd7f6b8531d046bc3ac40fce"
XHW_COMMIT = "eaaef1c32ba77b5f5a7e7fd4644d22e30af9b2ba"
XHW_CODE_SHA256 = "1dfedd30d43e4917995f4e0a852972a9d564c0d49f4881b720d4cf4e38dca0b0"
PRIMARY = X.BUNDLE / "results/FINAL_DEV" / FAMILY / f"seed_{SEED}"
PRIMARY_JOB = "1050357"
PRIMARY_GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
ROOT = X.BUNDLE / "hw_replicates"
UNIT = ROOT / "results/FINAL_DEV" / FAMILY / f"seed_{SEED}__L40S_replicate"
RELEASE = ROOT / "RELEASE_HWREP.json"
AGREEMENT = ROOT / f"AGREEMENT_seed_{SEED}.json"
CHANGED = ("scripts/v10_supplement/run.py",)
ADDED = ("scripts/v10_supplement/rulsif_v2_hwrep.py", "tests/v10/test_rulsif_v2_hwrep.py")
PRIMARY_FILES = ("IDENTITY.json", "COMPLETE.json", "HARDWARE.json", "DEV.jsonl", "CAL/step_0001.jsonl")
LABEL = {"kind": STATUS,
         "owner_words_verbatim": ["final 放开，CAL 留 L40S", "什么快什么来吧"],
         "statement": ("Post-hoc hardware sensitivity diagnostic decided after the seed-1001 final ran on RTX PRO 6000: the "
                       "same unit rerun on L40S. Not a fourth seed, not pooled with the primary results, never replaces "
                       "the primary seed-1001 result (the RTX run); reported only as a hardware-effect measurement.")}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def delta(base_files, current_files):
    changed = sorted(p for p in base_files if p in current_files and base_files[p] != current_files[p])
    added = sorted(set(current_files) - set(base_files))
    removed = sorted(set(base_files) - set(current_files))
    science = [p for p in changed + added + removed
               if p.startswith(("src/", "v10/supplement/", "slurm/")) or (p.startswith("tests/") and p not in ADDED)]
    return {"changed": changed, "added": added, "removed": removed,
            "allowlisted_only": set(changed) <= set(CHANGED) and set(added) <= set(ADDED) and not removed,
            "science_identical": not science, "science_files_touched": science}


def is_hwrep_release(path):
    try:
        return read(path).get("status") == STATUS
    except (OSError, ValueError):
        return False


def require_hwrep_release(path):
    require(bool(os.environ.get("SLURM_JOB_ID")), "Scientific execution requires Slurm")
    return verify_hwrep_release(path, os.environ.get("V10_SUPPLEMENT_EXPECT_RELEASE"))


def verify_hwrep_release(path, expected):
    path = Path(path).resolve()
    require(bool(expected) and sha(path) == expected, "Missing or changed submission-pinned diagnostic release")
    release = read(path)
    require(release.get("status") == STATUS and release.get("family") == FAMILY and release.get("seed") == SEED
            and release.get("phases") == ["final"] and tuple(release.get("gpu_names", ())) == GPU_NAMES,
            "Not the seed-1001 L40S hardware replicate release")
    inventory = source_inventory()
    code = digest(inventory)
    require(release.get("source_files") == inventory and release.get("code_sha256") == code, "Source identity changed")
    require(sha(XHW_RELEASE) == XHW_RELEASE_SHA256 and release.get("science_release") == {
        "path": str(XHW_RELEASE), "sha256": XHW_RELEASE_SHA256}, "RELEASE_XHW changed or not pinned")
    science = read(XHW_RELEASE)
    require(science.get("code_sha256") == XHW_CODE_SHA256 and science.get("source_commit") == XHW_COMMIT,
            "RELEASE_XHW names other code")
    change = delta(science["source_files"], inventory)
    require(change["allowlisted_only"] and change["science_identical"],
            "Source delta exceeds the diagnostic mode: " + repr(change))
    review = release.get("delta_review") or {}
    require(bool(review.get("path")) and sha(review["path"]) == review.get("sha256")
            and read(review["path"]).get("status") == "PASS" and read(review["path"]).get("new_code_sha256") == code,
            "Delta review missing or not PASS for this code")
    for name in PRIMARY_FILES:
        require(sha(PRIMARY / name) == release["primary_unit"]["files"][name], "Primary unit changed: " + name)
    primary = read(PRIMARY / "COMPLETE.json")
    require(primary.get("job") == PRIMARY_JOB and primary["identity"]["release_sha256"] == XHW_RELEASE_SHA256
            and read(PRIMARY / "HARDWARE.json").get("gpu_name") == PRIMARY_GPU, "Primary unit is not the RTX run")
    require(sha(X.BUNDLE / "results/CAL" / FAMILY / "SELECTION.json") == X.SELECTION_SHA256
            and release.get("cal_selection_sha256") == X.SELECTION_SHA256, "CAL selection changed")
    gate = science["l40s_gate"]
    require(release.get("l40s_gate") == gate and sha(gate["path"]) == gate["sha256"], "L40S gate changed")
    lock = release.get("data_lock") or {}
    require(lock == science.get("data_lock") and sha(lock["path"]) == lock.get("sha256"), "Data lock changed")
    require(release.get("replicate_unit") == str(UNIT) and UNIT.resolve() != PRIMARY.resolve()
            and not str(UNIT.resolve()).startswith(str((X.BUNDLE / "results").resolve())),
            "Replicate must live outside the primary results")
    policy_ref = release.get("scheduling_authorization") or {}
    policy_path = Path(policy_ref.get("path", ""))
    require(bool(policy_ref.get("path")) and sha(policy_path) == policy_ref.get("sha256"), "Scheduling authorization changed")
    policy = read(policy_path)
    active = read(X.PROJECT / "handoff/ACTIVE_SCHEDULING_POLICY.json")
    require(active.get("policy_id") == policy.get("policy_id")
            and Path(active.get("policy", "")).resolve() == policy_path.resolve()
            and policy.get("validated_supplement_overlap") is True, "Active policy does not authorize V10 work")
    require(not (X.PROJECT / "handoff/v10_parallel8_20261018/PAUSE_SUBMISSIONS").exists(), "V10 submissions paused")
    require(release.get("label") == LABEL, "Diagnostic label differs")
    return release


def prepare_replicate(ident, environment, release):
    """Exactly the primary unit's identity, in the separate replicate directory, on L40S; records hardware."""
    require(ident == read(PRIMARY / "IDENTITY.json"), "Replicate identity differs from the primary unit")
    require(UNIT.resolve() != PRIMARY.resolve(), "Replicate must never write the primary unit")
    require(environment.get("gpu_name") in GPU_NAMES, "Replicate runs on L40S only")
    existing = UNIT / "IDENTITY.json"
    if existing.exists():
        require(read(existing) == ident, "Replicate directory belongs to another unit")
    hardware = {"gpu_name": environment["gpu_name"], "gpu_capability": environment.get("gpu_capability"),
                "node": environment.get("host"), "slurm_nodelist": environment.get("slurm_nodelist"),
                "partition": environment.get("slurm_partition"), "gate_job": read(release["l40s_gate"]["path"])["job"]}
    if not (UNIT / "HARDWARE.json").exists():
        UNIT.mkdir(parents=True, exist_ok=True)
        write(UNIT / "HARDWARE.json", {**hardware, "first_job": environment.get("slurm_job_id"),
                                       "utc": time.strftime("%FT%TZ", time.gmtime())})
        write(UNIT / "DIAGNOSTIC.json", {**LABEL, "primary_unit": str(PRIMARY), "primary_job": PRIMARY_JOB,
                                         "primary_gpu": PRIMARY_GPU})
    return UNIT, hardware


def keyed(path):
    out = {}
    for row in rows(path):
        key = (row["core"], row["perturbation"])
        require(key not in out, "Duplicate registration row")
        out[key] = row
    return out


def compare(primary_rows, replicate_rows):
    require(set(primary_rows) == set(replicate_rows), "Registration sets differ")
    result = {"n": len(primary_rows)}
    for field in ("err_full", "theta_hat", "F_hat", "F_true", "rot_err_deg", "trans_err_full"):
        diffs = [abs(replicate_rows[k][field] - primary_rows[k][field]) for k in primary_rows]
        result[field] = {"identical_fraction": sum(d == 0 for d in diffs) / len(diffs), "max_abs_difference": max(diffs),
                         "median_abs_difference": statistics.median(diffs)}
    t = [max(abs(a - b) for a, b in zip(replicate_rows[k]["t_hat"], primary_rows[k]["t_hat"])) for k in primary_rows]
    result["t_hat"] = {"identical_fraction": sum(d == 0 for d in t) / len(t), "max_abs_difference": max(t)}
    flips = {name: sum((replicate_rows[k]["err_full"] < thr) != (primary_rows[k]["err_full"] < thr) for k in primary_rows)
             for name, thr in (("success42_flips", 42), ("success9_flips", 9))}
    result.update(flips)
    return result


def agreement():
    """Per-registration agreement of the L40S replicate with the primary RTX run (CAL8 and DEV32)."""
    rep_complete, pri_complete = read(UNIT / "COMPLETE.json"), read(PRIMARY / "COMPLETE.json")
    require(rep_complete["identity"] == pri_complete["identity"], "Units differ")
    dev_p, dev_r = keyed(PRIMARY / "DEV.jsonl"), keyed(UNIT / "DEV.jsonl")
    cal_p, cal_r = keyed(PRIMARY / "CAL/step_0001.jsonl"), keyed(UNIT / "CAL/step_0001.jsonl")
    def deltas(a, b):
        return {k: b[k] - a[k] for k in ("success42", "success9", "median_err")}
    result = {"schema": "v10.rulsif_exact_v2.hw_replicate_agreement.v1", "label": LABEL,
              "utc": time.strftime("%FT%TZ", time.gmtime()),
              "primary": {"unit": str(PRIMARY), "job": pri_complete["job"], "gpu": read(PRIMARY / "HARDWARE.json")["gpu_name"],
                          "complete_sha256": sha(PRIMARY / "COMPLETE.json")},
              "replicate": {"unit": str(UNIT), "job": rep_complete["job"], "gpu": read(UNIT / "HARDWARE.json")["gpu_name"],
                            "complete_sha256": sha(UNIT / "COMPLETE.json")},
              "same_identity": True,
              "fit": {"model_sha256_equal": read(PRIMARY / "checkpoints/step_0001.json")["model_sha256"]
                      == read(UNIT / "checkpoints/step_0001.json")["model_sha256"]},
              "DEV32": {"per_registration": compare(dev_p, dev_r), "primary": pri_complete["DEV32"],
                        "replicate": rep_complete["DEV32"], "replicate_minus_primary": deltas(pri_complete["DEV32"], rep_complete["DEV32"])},
              "DEV24": {"primary": pri_complete["DEV24"], "replicate": rep_complete["DEV24"],
                        "replicate_minus_primary": deltas(pri_complete["DEV24"], rep_complete["DEV24"])},
              "CAL8": {"per_registration": compare(cal_p, cal_r), "primary": metrics(list(cal_p.values())),
                       "replicate": metrics(list(cal_r.values())),
                       "replicate_minus_primary": deltas(metrics(list(cal_p.values())), metrics(list(cal_r.values())))}}
    write(AGREEMENT, result)
    return result


def write_agreement_after_run():
    """Called by run.py once the replicate is COMPLETE; a failure is recorded, never hides the unit."""
    try:
        return agreement()
    except Exception as error:   # the science unit is complete; the CLI can recompute the agreement
        write(ROOT / "AGREEMENT_ERROR.json", {"utc": time.strftime("%FT%TZ", time.gmtime()), "error": repr(error)})
        return None


# ------------------------------------------------------------------------------------------------ CLI (submits nothing)
def source_state():
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "scripts", "tests", "slurm",
                            "v10/supplement"], capture_output=True, text=True, check=True).stdout.strip()
    require(not dirty, "frozen tree has uncommitted scientific files")
    inventory = source_inventory()
    return commit, inventory, digest(inventory)


def ref(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def cli_delta_review(args):
    commit, inventory, code = source_state()
    change = delta(read(XHW_RELEASE)["source_files"], inventory)
    require(change["allowlisted_only"] and change["science_identical"], "delta exceeds the allowlist: " + repr(change))
    patch = ROOT / "DELTA_REVIEW.patch"
    require(not patch.exists() and not (ROOT / "DELTA_REVIEW.json").exists(), "delta review exists")
    ROOT.mkdir(parents=True, exist_ok=True)
    patch.write_text(subprocess.run(["git", "-C", str(REPO), "diff", XHW_COMMIT, commit, "--"] + change["changed"] + change["added"],
                                    capture_output=True, text=True, check=True).stdout)
    science = sorted(p for p in inventory if p.startswith(("src/", "v10/supplement/")))
    write(ROOT / "DELTA_REVIEW.json", {"schema": "v10.rulsif_exact_v2.hwrep_delta_review.v1", "status": "PASS",
          "base_commit": XHW_COMMIT, "base_code_sha256": XHW_CODE_SHA256, "base_release_sha256": XHW_RELEASE_SHA256,
          "new_commit": commit, "new_code_sha256": code, **{k: change[k] for k in ("changed", "added", "removed")},
          "science_files_byte_identical": len(science), "science_identical": True,
          "scope": "Diagnostic mode only: run.py dispatches a RELEASE_HWREP to the seed-1001 L40S replicate (same unit "
                   "identity as the primary, separate output root, agreement record). src/**, v10/supplement/*.json, "
                   "slurm/ and existing tests are byte-identical to RELEASE_XHW (eaaef1c3) and to 2e6ea58.",
          "patch": ref(patch), "utc": time.strftime("%FT%TZ", time.gmtime())})
    print(json.dumps({"new_commit": commit, "new_code_sha256": code, **{k: change[k] for k in ("changed", "added")}}, indent=1))


def cli_release(args):
    commit, inventory, code = source_state()
    science = read(XHW_RELEASE)
    pointer = read(X.PROJECT / "handoff/ACTIVE_SCHEDULING_POLICY.json")
    record = {"schema": "v10.supplement.release.rulsif_v2_hwrep.v1", "status": STATUS, "family": FAMILY, "seed": SEED,
              "phases": ["final"], "gpu_names": list(GPU_NAMES), "utc": time.strftime("%FT%TZ", time.gmtime()),
              "source_root": str(REPO.resolve()), "source_commit": commit, "source_files": inventory, "code_sha256": code,
              "science_release": {"path": str(XHW_RELEASE), "sha256": XHW_RELEASE_SHA256},
              "delta_review": ref(ROOT / "DELTA_REVIEW.json"), "cal_selection_sha256": X.SELECTION_SHA256,
              "primary_unit": {"path": str(PRIMARY), "job": PRIMARY_JOB, "gpu": PRIMARY_GPU,
                               "files": {name: sha(PRIMARY / name) for name in PRIMARY_FILES}},
              "replicate_unit": str(UNIT), "agreement_record": str(AGREEMENT), "l40s_gate": science["l40s_gate"],
              "data_lock": science["data_lock"], "scheduling_authorization": ref(pointer["policy"]), "label": LABEL,
              "reporting_rule": LABEL["statement"]}
    require(not RELEASE.exists(), "RELEASE_HWREP.json exists")
    write(RELEASE, record)
    pin = sha(RELEASE)
    verify_hwrep_release(RELEASE, pin)
    print(json.dumps({"release": str(RELEASE), "sha256": pin, "source_commit": commit, "code_sha256": code}, indent=1))
    print(f"V10_SUPPLEMENT_SOURCE={REPO.resolve()} V10_SUPPLEMENT_RELEASE={RELEASE} V10_SUPPLEMENT_EXPECT_RELEASE={pin} "
          f"sbatch --parsable --job-name=v10r2_hwrep1001 --partition=L40S --exclude=node51,node60 --time=03:00:00 "
          f"--export=ALL {REPO.resolve() / 'slurm/v10_supplement/rulsif_v2_run.sbatch'} final {FAMILY} --seed {SEED}")


def cli_preflight(args):
    import run as R   # record-level runner functions; no Torch import at module level
    pin = sha(RELEASE)
    release = verify_hwrep_release(RELEASE, pin)
    data_sha = sha(release["data_lock"]["path"])
    selection, dependency = R.checked_selection(X.BUNDLE / "results", FAMILY, X.BASE_RELEASE_SHA256, data_sha, require_exit=True)
    ident = R.identity(XHW_RELEASE_SHA256, FAMILY, selection["selected"]["candidate"], SEED, "FINAL_DEV", data_sha, dependency)
    print(json.dumps({"release_sha256": pin, "selection_sha256": dependency,
                      "identity_equals_primary": ident == read(PRIMARY / "IDENTITY.json"),
                      "replicate_unit": str(UNIT), "replicate_started": (UNIT / "IDENTITY.json").exists(),
                      "primary_job": PRIMARY_JOB, "primary_gpu": PRIMARY_GPU}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("delta-review", "release", "preflight", "agreement"):
        sub.add_parser(name)
    args = parser.parse_args()
    if args.command == "agreement":
        print(json.dumps(agreement(), indent=1, ensure_ascii=False)); return
    {"delta-review": cli_delta_review, "release": cli_release, "preflight": cli_preflight}[args.command](args)


if __name__ == "__main__":
    main()
