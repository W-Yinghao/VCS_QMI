"""Builder for the cross-hardware RuLSIF_exact_v2 final release (owner decision 「final 放开，CAL 留 L40S」).

Standard library only; reads records, never images; submits nothing (prints the commands). Run it
from the new frozen execution tree. Order: delta-review -> prepare-gpu-gates -> release; then the
main session submits the two gates and, once they PASS, the finals; `preflight` exercises the final's
record-level path (release, CAL-selection binding, seed state, gates) without Slurm or data.
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
sys.path.insert(0, str(REPO / "scripts/v10_supplement"))
import rulsif_v2_xhw as X  # noqa: E402
from v10.supplement.contract import (PARENT, RULSIF_V2_AMENDMENT, RULSIF_V2_RECORD, candidates,  # noqa: E402
                                     digest, read, schedule, sha, source_inventory)

SLURM = REPO / "slurm/v10_supplement"
EXCLUDE = "node51,node53,node60"
MANIFEST = X.XHW_DIR / "validation/MANIFEST.json"
REVIEW = X.XHW_DIR / "DELTA_REVIEW.json"
RELEASE = X.BUNDLE / "RELEASE_XHW.json"


def require(ok, message):
    if not ok:
        raise SystemExit("REFUSED: " + message)


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path)}


def write(path, value):
    path = Path(path)
    require(not path.exists(), f"{path} exists; never overwrite a pinned record")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def source_state():
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "scripts", "tests",
                            "slurm", "v10/supplement"], capture_output=True, text=True, check=True).stdout.strip()
    require(not dirty, "frozen tree has uncommitted scientific files:\n" + dirty)
    inventory = source_inventory()
    return commit, inventory, digest(inventory)


def delta_review(args):
    commit, inventory, code = source_state()
    base = read(X.BUNDLE / "RELEASE.json")
    require(sha(X.BUNDLE / "RELEASE.json") == X.BASE_RELEASE_SHA256, "CAL release changed")
    change = X.delta(base["source_files"], inventory)
    require(change["allowlisted_only"] and change["science_identical"], "delta exceeds the allowlist: " + repr(change))
    patch = subprocess.run(["git", "-C", str(REPO), "diff", X.BASE_COMMIT, commit, "--"] + change["changed"] + change["added"],
                           capture_output=True, text=True, check=True).stdout
    patch_path = X.XHW_DIR / "DELTA_REVIEW.patch"
    require(not patch_path.exists(), f"{patch_path} exists")
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(patch)
    science = sorted(p for p in inventory if p.startswith(("src/", "v10/supplement/")))
    review = {"schema": "v10.rulsif_exact_v2.xhw_delta_review.v1", "status": "PASS",
              "utc": time.strftime("%FT%TZ", time.gmtime()),
              "base_commit": X.BASE_COMMIT, "base_code_sha256": X.BASE_CODE_SHA256,
              "base_release_sha256": X.BASE_RELEASE_SHA256, "new_commit": commit, "new_code_sha256": code,
              "changed": change["changed"], "added": change["added"], "removed": change["removed"],
              "unchanged_files": change["unchanged"], "science_files_byte_identical": len(science),
              "science_identical": change["science_identical"],
              "scope": ("Only delta: final seeds accept " + ", ".join(X.FINAL_GPU_NAMES) + "; CAL stays " +
                        ", ".join(X.CAL_GPU_NAMES) + ". run.py: runtime_environment(gpu_names) and the "
                        "cross-hardware final path (pinned CAL selection consumed, CAL-release seeds refused, gate per "
                        "model, hardware recorded); validate.py: the gate reads accepted models from its pinned "
                        "manifest. Every other environment/precision check, recipe, data, search and selection rule "
                        "is unchanged; src/** and v10/supplement/*.json are byte-identical to the CAL release."),
              "patch": ref(patch_path)}
    write(REVIEW, review)
    print(json.dumps({k: review[k] for k in ("status", "new_commit", "new_code_sha256", "changed", "added",
                                             "science_files_byte_identical")}, indent=1))


def prepare_gpu_gates(args):
    commit, inventory, code = source_state()
    review = read(REVIEW)
    require(review.get("status") == "PASS" and review.get("new_code_sha256") == code, "delta review missing for this code")
    manifest = {"schema": "v10.supplement.validation.v1", "status": "VALIDATION_ONLY_NOT_A_SCIENTIFIC_RELEASE",
                "utc": time.strftime("%FT%TZ", time.gmtime()), "family": X.FAMILY,
                "purpose": "cross-hardware T01/execution gate for RuLSIF_exact_v2 final seeds (RTX PRO 6000, H100)",
                "accepted_gpu_names": list(X.FINAL_GPU_NAMES), "source_root": str(REPO.resolve()),
                "source_commit": commit, "source_files": inventory, "code_sha256": code,
                "data_lock": read(X.BUNDLE / "RELEASE.json")["data_lock"], "parent_protocol": ref(PARENT / "PROTOCOL.json"),
                "preregistration": ref(RULSIF_V2_RECORD), "amendment": ref(RULSIF_V2_AMENDMENT),
                "delta_review": ref(REVIEW), "base_release": {"path": str(X.BUNDLE / "RELEASE.json"),
                                                              "sha256": X.BASE_RELEASE_SHA256},
                "families": {X.FAMILY: {"candidate": candidates(X.FAMILY)[0], "schedule": schedule(X.FAMILY)}},
                "selection_rule": "First eligible candidate, seed 1117; validation outputs never enter CAL/final",
                "held_test_read": False, "dev_read": False,
                "gate_scope": "validate.py gate --family RuLSIF_exact_v2 on the allocated model: CUDA numerics, full FIT32 "
                              "fit, checkpoint resume/identity, FD geometry gradient, one complete common search on CAL core 0"}
    write(MANIFEST, manifest)
    pin = sha(MANIFEST)
    print(json.dumps({"validation_manifest": str(MANIFEST), "sha256": pin, "source_commit": commit, "code_sha256": code}, indent=1))
    for name, partition in (("rtx", "RTX6000PRO"), ("h100", "H100")):
        print(f"V10_VALIDATION_SOURCE={REPO.resolve()} V10_VALIDATION_MANIFEST={MANIFEST} V10_VALIDATION_SHA256={pin} "
              f"sbatch --parsable --job-name=v10r2_xgate_{name} --partition={partition} --exclude={EXCLUDE} "
              f"--time=00:30:00 --export=ALL {SLURM / 'rulsif_v2_gpu_gate.sbatch'}")


def release(args):
    commit, inventory, code = source_state()
    base = read(X.BUNDLE / "RELEASE.json")
    review = read(REVIEW)
    require(review.get("status") == "PASS" and review.get("new_code_sha256") == code, "delta review missing for this code")
    pointer = read(X.PROJECT / "handoff/ACTIVE_SCHEDULING_POLICY.json")
    selection = X.BUNDLE / "results/CAL" / X.FAMILY / "SELECTION.json"
    record = {"schema": "v10.supplement.release.rulsif_v2_xhw.v1", "status": X.XHW_RELEASE_STATUS, "family": X.FAMILY,
              "utc": time.strftime("%FT%TZ", time.gmtime()), "phases": ["final"],
              "source_root": str(REPO.resolve()), "source_commit": commit, "source_files": inventory, "code_sha256": code,
              "base_release": {"path": str(X.BUNDLE / "RELEASE.json"), "sha256": X.BASE_RELEASE_SHA256},
              "delta_review": ref(REVIEW), "parent_protocol_sha256": base["parent_protocol_sha256"],
              "families": {X.FAMILY: {"candidates": candidates(X.FAMILY), "schedule": schedule(X.FAMILY)}},
              "preregistration": ref(RULSIF_V2_RECORD), "amendment": ref(RULSIF_V2_AMENDMENT),
              "cal_selection": {"path": str(selection), "sha256": sha(selection)},
              "cal_exit": ref(X.BUNDLE / "exits" / f"{X.CAL_JOB}.json"),
              "cal_gpu_names": list(X.CAL_GPU_NAMES), "final_gpu_names": list(X.FINAL_GPU_NAMES),
              "l40s_gate": base["acceptance"]["GPU_GATE"], "xhw_validation_manifest": ref(MANIFEST),
              "gate_rule": ("A final on L40S is licensed by the CAL release gate (science code byte-identical); a final on "
                            "RTX PRO 6000 or H100 NVL needs a PASS validate.py gate on that model under the pinned "
                            "cross-hardware manifest with an exit-0 receipt, found at final start and recorded."),
              "seed_rule": "A seed whose final belongs to the CAL release 41313c39 stays valid and is refused here.",
              "hardware_rule": "One GPU model per final; GPU model, node, partition and allocated time recorded per final.",
              "data_lock": base["data_lock"], "scheduling_authorization": ref(pointer["policy"]),
              "results": base["results"], "owner_decision": X.OWNER_DECISION, "reporting_rule": X.REPORTING_RULE,
              "validation_outputs_excluded_from_science": True,
              "held_roles": {"TEST_REPEAT": "HOLD", "EPFL": "NOT_ACTIVATED", "CONFIRM": 0}}
    write(RELEASE, record)
    pin = sha(RELEASE)
    X.verify_xhw_release(RELEASE, pin)
    print(json.dumps({"release": str(RELEASE), "sha256": pin, "source_commit": commit, "code_sha256": code}, indent=1))
    print_finals(pin)


def print_finals(pin):
    env = (f"V10_SUPPLEMENT_SOURCE={REPO.resolve()} V10_SUPPLEMENT_RELEASE={RELEASE} V10_SUPPLEMENT_EXPECT_RELEASE={pin}")
    for seed in schedule(X.FAMILY)["final_seeds"]:
        print(f"{env} sbatch --parsable --job-name=v10r2_f{seed} --partition=RTX6000PRO,L40S,H100 --exclude={EXCLUDE} "
              f"--time=03:00:00 --export=ALL {SLURM / 'rulsif_v2_run.sbatch'} final {X.FAMILY} --seed {seed}")


def seed_state(seed):
    directory = X.BUNDLE / "results/FINAL_DEV" / X.FAMILY / f"seed_{seed}"
    identity = directory / "IDENTITY.json"
    if not identity.exists():
        return {"seed": seed, "state": "NOT_STARTED"}
    owner = read(identity).get("release_sha256")
    state = {"seed": seed, "release": "CAL release 41313c39" if owner == X.BASE_RELEASE_SHA256 else owner,
             "complete": (directory / "COMPLETE.json").exists()}
    if (directory / "HARDWARE.json").exists():
        state["hardware"] = read(directory / "HARDWARE.json")["gpu_name"]
    return state


def gates_now(release_record):
    found = {}
    for name in X.FINAL_GPU_NAMES:
        try:
            found[name] = X.gate_for(name, release_record)["gate_job"]
        except RuntimeError as error:
            found[name] = "NONE (" + str(error)[:60] + ")"
    return found


def preflight(args):
    import run as R  # runner functions only; no Torch import at module level
    pin = sha(RELEASE)
    release_record = X.verify_xhw_release(RELEASE, pin)
    data_sha = sha(release_record["data_lock"]["path"])
    root = X.BUNDLE / "results"
    selection, dependency = R.checked_selection(root, X.FAMILY, X.BASE_RELEASE_SHA256, data_sha, require_exit=True)
    require(dependency == X.SELECTION_SHA256, "selection hash differs")
    result = {"release_sha256": pin, "selection_sha256": dependency,
              "selected_candidate": selection["selected"]["candidate"], "gates_found": gates_now(release_record),
              "seeds": []}
    for seed in schedule(X.FAMILY)["final_seeds"]:
        ident = R.identity(pin, X.FAMILY, selection["selected"]["candidate"], seed, "FINAL_DEV", data_sha, dependency)
        state = seed_state(seed)
        state["runnable_here"] = state["state"] == "NOT_STARTED" if "state" in state else state["release"] == pin
        state["identity_sha256"] = digest(ident)
        result["seeds"].append(state)
    print(json.dumps(result, indent=1, ensure_ascii=False))


def status(args):
    pin = sha(RELEASE)
    release_record = X.verify_xhw_release(RELEASE, pin)
    print(json.dumps({"gates_found": gates_now(release_record),
                      "seeds": [seed_state(s) for s in schedule(X.FAMILY)["final_seeds"]]}, indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("delta-review", "prepare-gpu-gates", "release", "preflight", "status"):
        sub.add_parser(name)
    args = parser.parse_args()
    {"delta-review": delta_review, "prepare-gpu-gates": prepare_gpu_gates, "release": release,
     "preflight": preflight, "status": status}[args.command](args)


if __name__ == "__main__":
    main()
