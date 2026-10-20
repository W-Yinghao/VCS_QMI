"""Stage, freeze and print the submission commands of the D1-D3 bundle (CPU, records only).

  prepare_bundle.py stage     bundle directories, PROTOCOL.json (byte copy of the pre-registration),
                              D3 validation MANIFEST byte copy, verification of every frozen input
  prepare_bundle.py freeze    FREEZE.json: commit, sha256 of every diagnostics source file, frozen
                              inputs, job matrix; refuses on a dirty tree; written once
  prepare_bundle.py commands  verify the freeze exactly as the jobs do and print the sbatch commands

Never reads an image, never submits anything, never writes into a frozen tree or bundle.
"""
from __future__ import annotations

import argparse
import shlex
import shutil
from pathlib import Path
import sys

DIAG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DIAG / "src"))

from v10_sensitivity import common as C  # noqa: E402

SBATCH = DIAG / "slurm/v10_sensitivity/gpu.sbatch"
JOBS = (
    {"name": "v10d_d1_cal1017", "diagnostic": "d1", "args": ["CAL_1017"], "partition": "L40S", "time": "02:00:00"},
    {"name": "v10d_d1_f1001", "diagnostic": "d1", "args": ["FINAL_1001"], "partition": "L40S", "time": "02:00:00"},
    {"name": "v10d_d1_f1002", "diagnostic": "d1", "args": ["FINAL_1002"], "partition": "L40S", "time": "02:00:00"},
    {"name": "v10d_d1_f1003", "diagnostic": "d1", "args": ["FINAL_1003"], "partition": "L40S", "time": "02:00:00"},
    {"name": "v10d_d2_mindwtd", "diagnostic": "d2", "args": [], "partition": "L40S", "time": "02:00:00"},
    {"name": "v10d_d3_mlp8_rtx", "diagnostic": "d3", "args": ["MLP_VCS_perm8"], "partition": "RTX6000PRO", "time": "00:30:00"},
    {"name": "v10d_d3_infonce_rtx", "diagnostic": "d3", "args": ["InfoNCE_host[J_perm1]"], "partition": "RTX6000PRO", "time": "00:30:00"},
    {"name": "v10d_d3_mlp8_h100", "diagnostic": "d3", "args": ["MLP_VCS_perm8"], "partition": "H100", "time": "00:30:00"},
    {"name": "v10d_d3_infonce_h100", "diagnostic": "d3", "args": ["InfoNCE_host[J_perm1]"], "partition": "H100", "time": "00:30:00"},
)


def copy_once(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        C.require(target.read_bytes() == Path(source).read_bytes(), f"{target} exists and differs from {source}")
        return
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def verify_inputs(pre):
    inputs = pre["frozen_inputs"]
    checks = {}
    for key in ("data_lock", "parent_protocol"):
        C.require(C.sha(inputs[key]["path"]) == inputs[key]["sha256"], "Changed frozen input: " + key)
        checks[key] = "sha256 OK"
    comir = inputs["comir_release"]
    C.verify_release(comir)
    for name, unit in comir["units"].items():
        directory = Path(comir["path"]).parent / unit["dir"]
        C.require(C.sha(directory / "COMPLETE.json") == unit["complete_sha256"], "Changed CoMIR unit " + name)
        for step in pre["D1"]["steps"]:
            C.require((directory / "checkpoints" / f"step_{step:04d}.pt").exists(), f"{name}: checkpoint {step} missing")
    C.require(C.sha(Path(comir["path"]).parent / "results/CAL/CoMIR_native/SELECTION.json") == comir["cal_selection_sha256"],
              "Changed CoMIR CAL selection")
    checks["comir_release"] = "release, 38 source files, HEAD, 4 unit COMPLETE files, 48 pre-registered checkpoints OK"
    mind = inputs["mind_release"]
    C.verify_release(mind)
    C.require(C.sha(Path(mind["path"]).parent / mind["unit"]["dir"] / "COMPLETE.json") == mind["unit"]["complete_sha256"],
              "Changed MIND unit")
    checks["mind_release"] = "release, source files, HEAD, unit COMPLETE OK"
    manifest = inputs["pipeline_validation_manifest"]
    C.verify_manifest(manifest)
    for family, gate in manifest["accepted_l40s_gates"].items():
        C.require(C.sha(gate["path"]) == gate["sha256"], "Changed accepted gate " + family)
    checks["pipeline_validation_manifest"] = "manifest, source files, HEAD, accepted L40S gates OK"
    return checks


def stage():
    pre = C.prereg()
    for sub in ("D1", "D2", "D3/validation", "D3/summaries", "exits"):
        (C.BUNDLE / sub).mkdir(parents=True, exist_ok=True)
    copy_once(C.PREREGISTRATION, C.BUNDLE / "PROTOCOL.json")
    copy_once(pre["frozen_inputs"]["pipeline_validation_manifest"]["path"], C.BUNDLE / "D3/validation/MANIFEST.json")
    checks = verify_inputs(pre)
    record = {"status": "STAGED", "label": C.LABEL, "utc": C.utc(), "preregistration_sha256": C.sha(C.PREREGISTRATION),
              "protocol_copy_sha256": C.sha(C.BUNDLE / "PROTOCOL.json"),
              "d3_manifest_copy_sha256": C.sha(C.BUNDLE / "D3/validation/MANIFEST.json"), "input_checks": checks,
              "diag_head": C.git_head(C.REPO)}
    C.write(C.BUNDLE / "STAGED.json", record)
    print("STAGED", C.BUNDLE)


def freeze():
    pre = C.prereg()
    C.require(not C.dirty(), "Diagnostics tree has uncommitted or untracked changes:\n" + C.dirty())
    C.require(C.sha(C.BUNDLE / "PROTOCOL.json") == C.sha(C.PREREGISTRATION), "Stage the bundle first")
    C.require(C.sha(C.BUNDLE / "D3/validation/MANIFEST.json") == pre["frozen_inputs"]["pipeline_validation_manifest"]["sha256"],
              "Stage the bundle first")
    checks = verify_inputs(pre)
    files = C.source_hashes()
    record = {"schema": "v10.sensitivity.freeze.v1", "status": "FROZEN_FOR_SUBMISSION", "label": C.LABEL,
              "utc": C.utc(), "diagnostic_sources": {"root": str(C.REPO), "branch": "v10/sensitivity-diagnostics",
                                                     "commit": C.git_head(C.REPO), "files": files,
                                                     "code_sha256": C.digest(files)},
              "preregistration": {"path": str(C.PREREGISTRATION), "sha256": C.sha(C.PREREGISTRATION),
                                  "bundle_copy": str(C.BUNDLE / "PROTOCOL.json")},
              "frozen_inputs": pre["frozen_inputs"], "input_checks": checks,
              "outputs": {"bundle": str(C.BUNDLE), "rows": str(C.OUT_ROOT)},
              "jobs": [dict(job) for job in JOBS], "sbatch": str(SBATCH), "sbatch_sha256": C.sha(SBATCH),
              "submission": "main session only; this tool never submits"}
    C.write_once(C.BUNDLE / "FREEZE.json", record)
    print("FROZEN", C.sha(C.BUNDLE / "FREEZE.json"))


def commands():
    freeze_sha = C.sha(C.BUNDLE / "FREEZE.json")
    frozen = C.verify_freeze(freeze_sha)
    for job in frozen["jobs"]:
        argv = ["sbatch", "--parsable", f"--job-name={job['name']}", f"--partition={job['partition']}", "--qos=normal",
                "--exclude=node51,node60", "--nice=0", f"--time={job['time']}",
                f"--export=ALL,V10D_FREEZE_SHA256={freeze_sha}", frozen["sbatch"], job["diagnostic"], *job["args"]]
        print(" ".join(shlex.quote(a) for a in argv))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("stage", "freeze", "commands"))
    action = parser.parse_args().action
    {"stage": stage, "freeze": freeze, "commands": commands}[action]()


if __name__ == "__main__":
    main()
