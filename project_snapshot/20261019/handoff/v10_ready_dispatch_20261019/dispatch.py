"""Release each validated supplemental family and feed full CAL/final jobs.

Existing validation and BF1 controllers retain exclusive ownership of their jobs.
V9 independent closeout continues. No cancellation, held data, or blind retries.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from control import (POLICY_ID, capacity, checked_ref, command, digest, exit_receipt,
                     parsed_job, queue, read, ref, report_ok, require,
                     science_equivalence, sha, utc, write)

B = Path(__file__).resolve().parent
P = B.parents[1]


def active():
    return read(P/"handoff/ACTIVE_SCHEDULING_POLICY.json")["policy_id"] == POLICY_ID


def bundle():
    require(os.environ.get("SLURM_JOB_ID"), "Controller must execute under Slurm")
    require(sha(B/"FREEZE.json") == os.environ["V10_READY_FREEZE"], "Controller freeze changed")
    frozen = read(B/"FREEZE.json")
    for name, expected in frozen["files"].items():
        require(sha(B/name) == expected, "Frozen controller file changed: " + name)
    m = read(B/"MANIFEST.json")
    old = checked_ref(m["validation_manifest"])
    delta = science_equivalence(old["source_root"], m["source_root"],
                                old["source_files"], m["source_files"])
    require(delta == read(B/"DELTA_REVIEW.json"), "Scheduling migration differs from reviewed delta")
    checked_ref(m["data_lock"]); checked_ref(m["parent_protocol"])
    return m, old


def setup_imports(manifest):
    source = Path(manifest["source_root"])
    sys.path.insert(0, str(source/"src"))
    sys.path.insert(0, str(source/"scripts/v10_supplement"))
    import run as runner
    from v10.supplement import contract, records
    require(contract.source_inventory() == manifest["source_files"], "Imported wrong source inventory")
    return runner, contract, records


def preflight(m, old):
    source = Path(m["source_root"])
    env = dict(os.environ, V10_TEST_DEVICE="cpu", V10_READY_SOURCE=str(source),
               PYTHONPATH=str(source/"src")+":"+str(source))
    with (B/"CONTROL_TESTS.txt").open("w") as out:
        result = subprocess.run([sys.executable, str(B/"check_control.py")], env=env,
                                stdout=out, stderr=subprocess.STDOUT)
    require(result.returncode == 0, "Scheduling/release control tests failed")
    with (B/"CPU_REGRESSION.txt").open("w") as out:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                    "tests/v10/test_supplement.py", "tests/v10/test_v10_adapter.py",
                    "tests/v10/test_mf_vcs_core.py", "tests/v10/test_t00_manifest.py"],
                    cwd=source, env=env, stdout=out, stderr=subprocess.STDOUT)
    require(result.returncode == 0, "CPU regression failed on amended execution source")
    write(B/"PREFLIGHT.json", {"utc": utc(), "status": "PASS", "job": os.environ["SLURM_JOB_ID"],
          "code_sha256": m["code_sha256"], "control_tests": ref(B/"CONTROL_TESTS.txt"),
          "cpu_regression": ref(B/"CPU_REGRESSION.txt"), "scheduling_delta": ref(B/"DELTA_REVIEW.json"),
          "cuda_or_full_validation_claimed": False})
    # Reduce only this existing pending gate's positive nice offset. The same
    # job, source, resources and validation owner are retained. No running job
    # is preempted or cancelled, and administrator/user holds are untouched.
    ledger = read(Path(m["validation_manifest"]["path"]).parent/"LEDGER.json")
    gate = ledger["gate"].get("job")
    match = next((r for r in queue() if r["id"] == gate), None)
    if match and match["state"] == "PENDING" and match["name"] == "v10val_gate":
        r = subprocess.run(["scontrol", "update", "JobId="+gate, "Nice=0"],
                           capture_output=True, text=True, timeout=45)
        write(B/"GATE_PRIORITY.json", {"utc": utc(), "job": gate, "requested_nice": 0,
              "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr,
              "failure_is_nonfatal": True, "cancelled_jobs": []})


def matrix(contract):
    units = []
    for i, f in enumerate(contract.FAMILIES):
        units.append({"key": f"cal_{i}", "phase": "cal", "family": f, "index": i,
                      "status": "WAIT_VALIDATION"})
        units.extend({"key": f"final_{i}_{s}", "phase": "final", "family": f, "index": i,
                      "seed": s, "status": "WAIT_CAL"} for s in contract.schedule(f)["final_seeds"])
    return units


def validation_reports(family, rows, m, old, runner, contract, records):
    root = Path(m["validation_manifest"]["path"]).parent
    ledger = read(root/"LEDGER.json")
    gate = ledger["gate"]
    full = next(u for u in ledger["full"] if u["family"] == family)
    if gate["status"] != "COMPLETE_VERIFIED" or full["status"] != "COMPLETE_VERIFIED":
        return None, {"gate": gate["status"], "full": full["status"]}
    live = {r["id"] for r in rows}
    require(gate["job"] not in live and full["job"] not in live, "Validation still live despite COMPLETE")
    vh = m["validation_manifest"]["sha256"]
    for unit in (gate, full):
        exit_receipt(root/"exits"/(unit["job"]+".json"), unit["job"], vh, "validation_manifest_sha256")
    aggregate = checked_ref(gate["reports"]["GATE.json"])
    require(aggregate["status"] == "PASS" and aggregate["code_sha256"] == old["code_sha256"]
            and aggregate["validation_manifest_sha256"] == vh and aggregate["job"] == gate["job"]
            and {r["family"] for r in aggregate["families"]} == set(contract.FAMILIES),
            "Incomplete or mismatched CUDA aggregate")
    gate_ref = next(r for r in aggregate["families"] if r["family"] == family)
    checked_ref(gate_ref)
    refs = {"T01": {k: gate_ref[k] for k in ("path", "sha256")},
            "T02": full["reports"]["T02.json"], "RUNNER": full["reports"]["RUNNER.json"]}
    reports = {}
    for stage, record in refs.items():
        checked_ref(record)
        reports[stage] = report_ok(record["path"], family, old["code_sha256"], vh,
                                   gate["job"] if stage == "T01" else full["job"])
    ident = runner.identity(vh, family, contract.candidates(family)[0], 1117,
                            "VALIDATION_FULL_CAL", m["data_lock"]["sha256"])
    complete = root/"runs"/full["job"]/"unit/COMPLETE.json"
    result = records.verified_complete(complete.parent, ident)
    require(result is not None, "Full validation result is missing")
    expected_n = len(contract.schedule(family)["cal_steps"])
    for stage in ("T02", "RUNNER"):
        r = reports[stage]
        require(Path(r["full_complete"]).resolve() == complete.resolve()
                and r["full_complete_sha256"] == sha(complete)
                and r["full_training_steps"] == contract.schedule(family)["steps"]
                and r["registrations"] == 80*expected_n
                and r["cal_checkpoints"] == expected_n, "Incomplete validation coverage")
    require(all(c["metrics"]["n"] == 80 for c in result["cal_checkpoints"]), "Incomplete CAL blocks")
    require({r["role"] for r in read(complete.parent/"DATA_READS.json")} <= {"FIT", "CAL"},
            "Full validation read an unapproved role")
    return {"reports": refs, "gate_exit": ref(root/"exits"/(gate["job"]+".json")),
            "full_exit": ref(root/"exits"/(full["job"]+".json")), "complete": ref(complete)}, None


def release_family(unit, proofs, m, old, contract):
    directory = B/"families"/str(unit["index"])
    path = directory/"RELEASE.json"
    require(not path.exists(), "Existing release without ledger ownership; reconcile before reuse")
    family = unit["family"]
    acceptance = {}
    for stage, original in proofs["reports"].items():
        target = directory/"acceptance"/(stage+".json")
        write(target, {"utc": utc(), "status": "PASS", "stage": stage, "family": family,
              "code_sha256": m["code_sha256"], "tested_gpu_code_sha256": old["code_sha256"],
              "acceptance_mode": "BYTE_IDENTICAL_SCIENCE_WITH_TESTED_SCHEDULING_AMENDMENT",
              "description": "GPU/full tests ran on the original frozen validation source. Training, numerical, data, search and runner files are byte-identical; only require_release scheduling authorization changed. This is explicit acceptance transfer, not a claim of rerunning those GPU tests on a new commit.",
              "original_report": original, "validation_evidence": proofs,
              "delta_review": ref(B/"DELTA_REVIEW.json"), "cpu_and_control_regression": ref(B/"PREFLIGHT.json"),
              "method_review": ref(B/"METHOD_REVIEW.json")})
        acceptance[stage] = ref(target)
    release = {"schema": "v10.supplement.release.v1", "status": "FROZEN_VALIDATED", "utc": utc(),
               "family": family, "source_root": m["source_root"], "source_commit": m["source_commit"],
               "source_files": m["source_files"], "code_sha256": m["code_sha256"],
               "parent_protocol_sha256": m["parent_protocol"]["sha256"],
               "families": {family: {"candidates": contract.candidates(family), "schedule": contract.schedule(family)}},
               "acceptance": {family: acceptance}, "data_lock": m["data_lock"],
               "comir_source": m["comir_source"], "scheduling_authorization": ref(B/"POLICY.json"),
               "method_review": ref(B/"METHOD_REVIEW.json"), "validation_outputs_excluded_from_science": True,
               "held_roles": {"TEST_REPEAT": "HOLD", "EPFL": "NOT_ACTIVATED", "CONFIRM": 0}}
    write(path, release)
    previous = os.environ.get("V10_SUPPLEMENT_EXPECT_RELEASE")
    os.environ["V10_SUPPLEMENT_EXPECT_RELEASE"] = sha(path)
    try:
        contract.require_release(path, family)
    finally:
        if previous is None: os.environ.pop("V10_SUPPLEMENT_EXPECT_RELEASE", None)
        else: os.environ["V10_SUPPLEMENT_EXPECT_RELEASE"] = previous
    return ref(path)


def observe(unit, rows, m, runner, records, ledger):
    if unit["status"] != "SUBMITTED" or any(r["id"] == unit["job"] for r in rows):
        return
    unit.setdefault("absent_unix", time.time())
    release = unit["release"]
    base = Path(release["path"]).parent
    receipt = base/"exits"/(unit["job"]+".json")
    if not receipt.exists() and time.time()-unit["absent_unix"] < 300:
        return
    try:
        exit_receipt(receipt, unit["job"], release["sha256"], "release_sha256")
        checked_ref(release)
        require(not (base/"failures"/(unit["job"]+".json")).exists(), "Formal failure record exists")
        ends = [r for r in records.rows(base/"COST_LEDGER.jsonl") if r["job"] == unit["job"] and r["event"] == "END"]
        require(len(ends) == 1 and ends[0]["status"] == "COMPLETE", "Formal END not complete")
        root = base/"results"
        selection, dep = runner.checked_selection(root, unit["family"], release["sha256"],
                                                   m["data_lock"]["sha256"], require_exit=True)
        if unit["phase"] == "cal":
            require(selection["job"] == unit["job"], "CAL selection belongs to another job")
            unit["selection"] = ref(root/"CAL"/unit["family"]/"SELECTION.json")
        else:
            ident = runner.identity(release["sha256"], unit["family"], selection["selected"]["candidate"],
                                    unit["seed"], "FINAL_DEV", m["data_lock"]["sha256"], dep)
            directory = root/"FINAL_DEV"/unit["family"]/f"seed_{unit['seed']}"
            result = records.verified_complete(directory, ident)
            require(result is not None and result["job"] == unit["job"]
                    and result["DEV32"]["n"] == 320 and result["DEV24"]["n"] == 240,
                    "Final result or DEV coverage incomplete")
            unit["complete"] = ref(directory/"COMPLETE.json")
        unit.update(status="COMPLETE_VERIFIED", exit=ref(receipt), completed_utc=utc())
    except Exception as error:
        unit.update(status="FAILED_REVIEW_REQUIRED", error=repr(error))
    write(B/"LEDGER.json", ledger)


def submit(unit, m, ledger):
    require(unit["status"] == "READY" and not unit.get("job"), "Duplicate submission")
    require(active(), "Scheduling policy superseded")
    checked_ref(unit["release"])
    source = Path(m["source_root"])
    name = "v10s_c"+str(unit["index"]) if unit["phase"] == "cal" else f"v10s_f{unit['index']}_{unit['seed']}"
    args = ["sbatch", "--parsable", "--job-name="+name,
            "--export=ALL,V10_SUPPLEMENT_SOURCE="+str(source)+",V10_SUPPLEMENT_RELEASE="+unit["release"]["path"]+
            ",V10_SUPPLEMENT_EXPECT_RELEASE="+unit["release"]["sha256"],
            str(source/"slurm/v10_supplement/run.sbatch"), unit["phase"], unit["family"]]
    if unit["phase"] == "final": args.extend(["--seed", str(unit["seed"])])
    unit.update(status="SUBMIT_INTENT", command=args, submit_utc=utc())
    write(B/"LEDGER.json", ledger)
    r = subprocess.run(args, capture_output=True, text=True, timeout=45)
    unit["submission"] = {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    write(B/"LEDGER.json", ledger)
    if r.returncode != 0 and not r.stdout.strip() and "QOSMaxSubmitJobPerUserLimit" in r.stderr:
        unit["status"] = "READY"; write(B/"LEDGER.json", ledger); return
    unit.update(status="SUBMITTED", job=parsed_job(r))
    write(B/"LEDGER.json", ledger)
    print("SUBMITTED", unit["key"], unit["job"], flush=True)


def main():
    m, old = bundle()
    runner, contract, records = setup_imports(m)
    with (B/"DISPATCH.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (B/"LEDGER.json").exists(), "Existing ledger; do not duplicate this controller")
        require(active(), "Policy is not active")
        write(B/"STATUS.json", {"utc": utc(), "status": "CPU_PREFLIGHT", "controller": os.environ["SLURM_JOB_ID"]})
        preflight(m, old)
        ledger = {"utc": utc(), "controller": os.environ["SLURM_JOB_ID"], "policy_id": POLICY_ID,
                  "units": matrix(contract), "status": "WAIT_METHOD_VALIDATION"}
        write(B/"LEDGER.json", ledger)
        validation_root = Path(m["validation_manifest"]["path"]).parent
        for _ in range(94*120):
            if not active():
                write(B/"STATUS.json", {"utc": utc(), "status": "SUPERSEDED_NO_NEW_SUBMISSIONS"}); return
            try:
                rows = queue()
            except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                write(B/"STATUS.json", {"utc": utc(), "status": "QUEUE_RETRY_NO_SUBMISSION", "error": repr(error)})
                time.sleep(30); continue
            bundle()
            waits = {}
            for unit in ledger["units"]:
                observe(unit, rows, m, runner, records, ledger)
                if unit["status"] == "WAIT_VALIDATION":
                    try:
                        proofs, reason = validation_reports(unit["family"], rows, m, old, runner, contract, records)
                        if proofs:
                            unit.update(release=release_family(unit, proofs, m, old, contract), status="READY")
                            write(B/"LEDGER.json", ledger)
                        else: waits[unit["family"]] = reason
                    except Exception as error:
                        unit.update(status="FAILED_REVIEW_REQUIRED", error=repr(error))
                        write(B/"LEDGER.json", ledger)
                if unit["status"] == "WAIT_CAL":
                    cal = next(u for u in ledger["units"] if u["key"] == "cal_"+str(unit["index"]))
                    if cal["status"] == "COMPLETE_VERIFIED":
                        unit.update(status="READY", release=cal["release"], selection=cal["selection"])
                    elif cal["status"] == "FAILED_REVIEW_REQUIRED":
                        unit.update(status="BLOCKED_FAILED_CAL", error="Upstream CAL/validation failed")
            validation = read(validation_root/"LEDGER.json")
            # Existing validation owns its seven submissions. Let it finish
            # enqueueing before this controller competes for the pending cap.
            validation_submissions_finished = all(u["status"] not in ("PLANNED", "SUBMIT_INTENT")
                                                  for u in validation["full"])
            if validation_submissions_finished:
                for unit in ledger["units"]:
                    if unit["status"] == "READY":
                        try: free = capacity(queue())
                        except (RuntimeError, OSError, subprocess.SubprocessError): break
                        if not free: break
                        submit(unit, m, ledger)
            done = sum(u["status"] == "COMPLETE_VERIFIED" for u in ledger["units"])
            failed = [u["key"] for u in ledger["units"] if u["status"] in ("FAILED_REVIEW_REQUIRED", "BLOCKED_FAILED_CAL")]
            ledger["status"] = "ACTIVE" if any(u.get("release") for u in ledger["units"]) else "WAIT_METHOD_VALIDATION"
            write(B/"LEDGER.json", ledger)
            write(B/"STATUS.json", {"utc": utc(), "status": ledger["status"], "controller": ledger["controller"],
                  "formal_jobs_complete": done, "formal_jobs_planned": len(ledger["units"]),
                  "formal_jobs_submitted": sum(bool(u.get("job")) for u in ledger["units"]),
                  "cal_candidates": 19, "final_units": 15, "failed": failed, "validation_waits": waits,
                  "v9_completion_dependency": False, "validation_submissions_finished": validation_submissions_finished,
                  "shared_normal_gpu_cap": 8, "pending_target": 8, "normal_job_cap": 29,
                  "cancelled_running_jobs": []})
            if done+len(failed) == len(ledger["units"]):
                write(B/"FINISHED.json", {**ledger, "status": "PASS" if not failed else "REVIEW_REQUIRED"})
                require(not failed, "Formal failures require review: " + repr(failed)); return
            time.sleep(30)
        raise RuntimeError("Supplement controller reached 94-hour limit")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write(B/"ERROR.json", {"utc": utc(), "status": "FAIL", "error": repr(error),
                               "traceback": traceback.format_exc()})
        raise

