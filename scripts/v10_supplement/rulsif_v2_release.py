"""RuLSIF_exact_v2 bundle builder: GPU-gate manifest and the minimal explicit release.

Standard library only; reads records, never images; submits nothing (prints the commands).
Run it from the frozen execution worktree that the GPU jobs will use (its path becomes
source_root). Order: FIT gate PASS (CPU job) -> prepare-gpu-gate -> GPU gate PASS -> release.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v10.supplement.contract import (PARENT, PROJECT, RULSIF_V2_AMENDMENT, RULSIF_V2_RECORD,  # noqa: E402
                                     RULSIF_V2_RELEASE_STATUS, candidates, check_fit_gate_v2, digest, read,
                                     rulsif_v2_amendment, schedule, sha, source_inventory, verify_release_v2)

FAMILY = "RuLSIF_exact_v2"
BUNDLE = PROJECT / "handoff/v10_rulsif_v2_20261019"
OUTPUTS = Path("/projects/EEG-foundation-model/yinghao/outputs_v10/rulsif_v2")
DATA_LOCK = PROJECT / "handoff/v10_bf1_20261017/DATA_LOCK.json"
DATA_LOCK_SHA256 = "e65c60a33f095933e673613fc58f2754485933524259e548c6a0f5dc380f72c5"
POLICY_POINTER = PROJECT / "handoff/ACTIVE_SCHEDULING_POLICY.json"
SLURM = REPO / "slurm/v10_supplement"


def require(ok, message):
    if not ok:
        raise SystemExit("REFUSED: " + message)


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def source_state():
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "scripts",
                            "tests", "slurm", "v10/supplement"], capture_output=True, text=True,
                           check=True).stdout.strip()
    require(not dirty, "execution worktree has uncommitted scientific files:\n" + dirty)
    inventory = source_inventory()
    return commit, inventory, digest(inventory)


def checked_fit_gate(path, code):
    report = read(path)
    require(report.get("stage") == "FIT_FUNCTIONAL_GATE" and report.get("status") == "PASS"
            and report.get("family") == FAMILY and report.get("code_sha256") == code
            and report.get("preregistration_sha256") == sha(RULSIF_V2_RECORD)
            and report.get("held_test_read") is False and report.get("dev_read") is False
            and report.get("cal_read") is False and report.get("roles_read") == ["FIT"],
            "FIT functional gate is not PASS for this exact code and record")
    try:   # same coverage rule as the Slurm-side release check (amendment-aware)
        check_fit_gate_v2(report, sha(RULSIF_V2_RECORD))
    except RuntimeError as error:
        require(False, str(error))
    require(sha(report["rows"]) == report["rows_sha256"], "FIT gate rows changed")
    return report


def prepare_gpu_gate(args):
    commit, inventory, code = source_state()
    checked_fit_gate(args.fit_gate, code)
    require(sha(DATA_LOCK) == DATA_LOCK_SHA256, "data lock changed")
    manifest = {"schema": "v10.supplement.validation.v1", "status": "VALIDATION_ONLY_NOT_A_SCIENTIFIC_RELEASE",
                "utc": time.strftime("%FT%TZ", time.gmtime()), "family": FAMILY,
                "source_root": str(REPO.resolve()), "source_commit": commit, "source_files": inventory,
                "code_sha256": code, "data_lock": ref(DATA_LOCK), "parent_protocol": ref(PARENT / "PROTOCOL.json"),
                "preregistration": ref(RULSIF_V2_RECORD), "fit_gate": ref(args.fit_gate),
                "amendment": ref(RULSIF_V2_AMENDMENT) if rulsif_v2_amendment() is not None else None,
                "families": {FAMILY: {"candidate": candidates(FAMILY)[0], "schedule": schedule(FAMILY)}},
                "selection_rule": "First pre-registered candidate, seed 1117; validation outputs never enter CAL/final",
                "held_test_read": False, "dev_read": False,
                "gate_scope": "validate.py gate --family RuLSIF_exact_v2: CUDA numerics (incl. test_rulsif_v2.py), "
                              "full FIT32 fit, checkpoint resume/identity, one complete common search on CAL core 0"}
    path = BUNDLE / "validation/MANIFEST.json"
    require(not path.exists() or args.force_new_manifest, "validation manifest exists; refusing to overwrite")
    write(path, manifest)
    pin = sha(path)
    print(json.dumps({"validation_manifest": str(path), "sha256": pin, "source_commit": commit, "code_sha256": code}, indent=1))
    print("\n# GPU gate (L40S, nice 0; ~1-3 min of GPU work, limit 45 min):")
    print(f"V10_VALIDATION_SOURCE={REPO.resolve()} V10_VALIDATION_MANIFEST={path} V10_VALIDATION_SHA256={pin} \\\n"
          f"  sbatch --parsable --export=ALL {SLURM / 'rulsif_v2_gpu_gate.sbatch'}")


def release(args):
    commit, inventory, code = source_state()
    fit_report = checked_fit_gate(args.fit_gate, code)
    manifest_path = BUNDLE / "validation/MANIFEST.json"
    manifest = read(manifest_path)
    require(manifest.get("code_sha256") == code and manifest.get("source_root") == str(REPO.resolve()),
            "GPU-gate manifest names other code or another source root")
    job = str(args.gpu_gate_job)
    receipt = read(BUNDLE / "validation/exits" / f"{job}.json")
    require(receipt.get("job") == job and receipt.get("exit_code") == 0
            and receipt.get("validation_manifest_sha256") == sha(manifest_path), "GPU gate exit receipt not successful")
    aggregate = read(BUNDLE / "validation/runs" / job / "GATE.json")
    require(aggregate.get("status") == "PASS" and aggregate.get("requested_families") == [FAMILY]
            and aggregate.get("code_sha256") == code and aggregate.get("job") == job, "GPU gate aggregate not PASS")
    family_ref = aggregate["families"][0]
    require(family_ref["family"] == FAMILY and sha(family_ref["path"]) == family_ref["sha256"], "GPU gate report changed")
    gpu = read(family_ref["path"])
    require(gpu.get("status") == "PASS" and gpu.get("stage") == "T01_AND_EXECUTION_GATE" and gpu.get("code_sha256") == code
            and gpu.get("environment", {}).get("gpu_name") == "NVIDIA L40S", "GPU gate report not PASS on L40S")
    pointer = read(POLICY_POINTER)
    policy_path = Path(pointer["policy"])
    results = BUNDLE / "results"
    OUTPUTS.joinpath("results").mkdir(parents=True, exist_ok=True)
    if not results.exists():
        results.symlink_to(OUTPUTS / "results", target_is_directory=True)
    require(results.resolve() == (OUTPUTS / "results").resolve(), "results path is not the outputs directory")
    record = {"schema": "v10.supplement.release.rulsif_v2.v1", "status": RULSIF_V2_RELEASE_STATUS, "family": FAMILY,
              "utc": time.strftime("%FT%TZ", time.gmtime()), "source_root": str(REPO.resolve()), "source_commit": commit,
              "source_files": inventory, "code_sha256": code, "parent_protocol_sha256": sha(PARENT / "PROTOCOL.json"),
              "families": {FAMILY: {"candidates": candidates(FAMILY), "schedule": schedule(FAMILY)}},
              "preregistration": ref(RULSIF_V2_RECORD),
              "amendment": ref(RULSIF_V2_AMENDMENT) if rulsif_v2_amendment() is not None else None,
              "acceptance": {"FIT_GATE": ref(args.fit_gate), "GPU_GATE": {k: family_ref[k] for k in ("path", "sha256")}},
              "gpu_gate_exit": ref(BUNDLE / "validation/exits" / f"{job}.json"),
              "fit_gate_status": {"v2": fit_report["families"][FAMILY]["status"],
                                  "v1_same_gate": fit_report["families"].get("RuLSIF_exact", {}).get("status")},
              "end_to_end_acceptance": "carried by the first formal CAL candidate (run.py identity, complete-artifact "
                                       "and checkpoint checks); no separate seed-1117 full-length validation",
              "data_lock": ref(DATA_LOCK), "scheduling_authorization": ref(policy_path),
              "results": {"link": str(results), "target": str((OUTPUTS / "results").resolve())},
              "budget": {"cal": "<= 9 candidates and 8 L40S allocated GPU-h (runner-enforced)", "final_seeds": [1001, 1002, 1003]},
              "validation_outputs_excluded_from_science": True,
              "held_roles": {"TEST_REPEAT": "HOLD", "EPFL": "NOT_ACTIVATED", "CONFIRM": 0}}
    path = BUNDLE / "RELEASE.json"
    require(not path.exists(), "RELEASE.json exists; never overwrite a release that may own results")
    write(path, record)
    pin = sha(path)
    verify_release_v2(path, pin)
    print(json.dumps({"release": str(path), "sha256": pin, "source_commit": commit, "code_sha256": code}, indent=1))
    env = f"V10_SUPPLEMENT_SOURCE={REPO.resolve()} V10_SUPPLEMENT_RELEASE={path} V10_SUPPLEMENT_EXPECT_RELEASE={pin}"
    run = SLURM / "rulsif_v2_run.sbatch"
    print(f"\n# CAL ({len(candidates(FAMILY))} eligible candidates sequential, seed 1017, 8 L40S GPU-h cap):")
    print(f"CAL=$({env} sbatch --parsable --job-name=v10r2_cal --time=06:00:00 --export=ALL {run} cal {FAMILY})")
    print("\n# Final seeds (each ~1.2 GPU-h; afterok: start only after a successful CAL exit):")
    for seed in schedule(FAMILY)["final_seeds"]:
        print(f"{env} sbatch --parsable --job-name=v10r2_f{seed} --time=03:00:00 --dependency=afterok:$CAL "
              f"--export=ALL {run} final {FAMILY} --seed {seed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-gpu-gate"); p.add_argument("--fit-gate", type=Path, required=True)
    p.add_argument("--force-new-manifest", action="store_true")
    r = sub.add_parser("release"); r.add_argument("--fit-gate", type=Path, required=True)
    r.add_argument("--gpu-gate-job", required=True)
    args = parser.parse_args()
    {"prepare-gpu-gate": prepare_gpu_gate, "release": release}[args.command](args)


if __name__ == "__main__":
    main()
