"""Independent per-method gates -> full validation -> formal CAL -> final seeds.

Owns all new-method submissions; failures block only the dependent family.
Existing BF1/V9 jobs continue. No cancellation, held data, or blind retries.
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
    require(sha(B/"FREEZE.json") == os.environ["V10_NATIVE_RECOVERY_FREEZE"], "Controller freeze changed")
    frozen = read(B/"FREEZE.json")
    for name, expected in frozen["files"].items():
        require(sha(B/name) == expected, "Frozen controller file changed: " + name)
    m = read(B/"MANIFEST.json")
    old = checked_ref(m["validation_manifest"])
    require(old["source_root"] == m["source_root"] and old["source_files"] == m["source_files"]
            and old["code_sha256"] == m["code_sha256"] == digest(m["source_files"]),
            "Validation and formal execution must use this same source version")
    for name, expected in m["source_files"].items():
        require(sha(Path(m["source_root"])/name) == expected, "Scientific source changed: " + name)
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
          "cpu_regression": ref(B/"CPU_REGRESSION.txt"), "repair_record": ref(B/"REPAIR.json"),
          "cuda_or_full_validation_claimed": False})


def matrix(contract, indices=None):
    units = []
    for i, f in enumerate(contract.FAMILIES):
        if indices is not None and i not in indices:
            continue
        units.append({"key": f"gate_{i}", "phase": "gate", "family": f, "index": i,
                      "status": "READY"})
        units.append({"key": f"full_{i}", "phase": "full", "family": f, "index": i,
                      "status": "WAIT_GATE"})
        units.append({"key": f"cal_{i}", "phase": "cal", "family": f, "index": i,
                      "status": "WAIT_FULL"})
        units.extend({"key": f"final_{i}_{s}", "phase": "final", "family": f, "index": i,
                      "seed": s, "status": "WAIT_CAL"} for s in contract.schedule(f)["final_seeds"])
    return units


def validation_reports(family, rows, m, old, runner, contract, records, ledger):
    root = Path(m["validation_manifest"]["path"]).parent
    gate = next(u for u in ledger["units"] if u["phase"] == "gate" and u["family"] == family)
    full = next(u for u in ledger["units"] if u["phase"] == "full" and u["family"] == family)
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
            and {r["family"] for r in aggregate["families"]} == {family}
            and aggregate["requested_families"] == [family],
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
    # These are the actual same-code validation reports, not transferred PASS
    # records. The failed original version supplies no acceptance evidence.
    acceptance = proofs["reports"]
    release = {"schema": "v10.supplement.release.v1", "status": "FROZEN_VALIDATED", "utc": utc(),
               "family": family, "source_root": m["source_root"], "source_commit": m["source_commit"],
               "source_files": m["source_files"], "code_sha256": m["code_sha256"],
               "parent_protocol_sha256": m["parent_protocol"]["sha256"],
               "families": {family: {"candidates": contract.candidates(family), "schedule": contract.schedule(family)}},
               "acceptance": {family: acceptance}, "data_lock": m["data_lock"],
               "comir_source": m["comir_source"], "scheduling_authorization": m["policy"],
               "method_review": ref(B/"METHOD_REVIEW.json"), "validation_evidence": proofs,
               "cpu_and_control_regression": ref(B/"PREFLIGHT.json"),
               "validation_outputs_excluded_from_science": True,
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
    if unit["phase"] in ("gate", "full"):
        root = Path(m["validation_manifest"]["path"]).parent
        receipt = root/"exits"/(unit["job"]+".json")
        if not receipt.exists() and time.time()-unit["absent_unix"] < 300:
            return
        try:
            vh = m["validation_manifest"]["sha256"]
            exit_receipt(receipt, unit["job"], vh, "validation_manifest_sha256")
            names = ["GATE.json"] if unit["phase"] == "gate" else ["T02.json", "RUNNER.json"]
            reports = {}
            for name in names:
                path = root/"runs"/unit["job"]/name
                report = read(path)
                if unit["phase"] == "gate":
                    require(report.get("status") == "PASS" and report.get("job") == unit["job"]
                            and report.get("code_sha256") == m["code_sha256"]
                            and report.get("validation_manifest_sha256") == vh
                            and report.get("requested_families") == [unit["family"]]
                            and len(report.get("families", [])) == 1, "Invalid per-family CUDA gate")
                    family_ref = report["families"][0]
                    require(family_ref["family"] == unit["family"], "CUDA gate family mismatch")
                    checked_ref(family_ref)
                    report_ok(family_ref["path"], unit["family"], m["code_sha256"], vh, unit["job"])
                else:
                    report_ok(path, unit["family"], m["code_sha256"], vh, unit["job"])
                reports[name] = ref(path)
            unit.update(status="COMPLETE_VERIFIED", reports=reports, exit=ref(receipt), completed_utc=utc())
        except Exception as error:
            unit.update(status="FAILED_REVIEW_REQUIRED", error=repr(error), failed_utc=utc())
        write(B/"LEDGER.json", ledger)
        return
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
    source = Path(m["source_root"])
    if unit["phase"] in ("gate", "full"):
        name = "v10g_"+unit["phase"]+str(unit["index"])
        args = ["sbatch", "--parsable", "--job-name="+name, "--nice=0",
                "--export=ALL,V10_VALIDATION_SOURCE="+str(source)+",V10_VALIDATION_MANIFEST="+
                m["validation_manifest"]["path"]+",V10_VALIDATION_SHA256="+m["validation_manifest"]["sha256"],
                str(source/"slurm/v10_supplement/validate.sbatch"), unit["phase"], "--family", unit["family"]]
    else:
        checked_ref(unit["release"])
        name = "v10g_cal"+str(unit["index"]) if unit["phase"] == "cal" else f"v10g_f{unit['index']}_{unit['seed']}"
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


def advance_waiters(ledger):
    by_key = {u["key"]: u for u in ledger["units"]}
    dependencies = {"WAIT_GATE": ("gate_", "READY"), "WAIT_FULL": ("full_", "READY_TO_RELEASE"),
                    "WAIT_CAL": ("cal_", "READY")}
    for unit in ledger["units"]:
        if unit["status"] not in dependencies:
            continue
        prefix, ready = dependencies[unit["status"]]
        parent = by_key[prefix+str(unit["index"])]
        if parent["status"] in ("FAILED_REVIEW_REQUIRED", "BLOCKED_UPSTREAM_FAILURE"):
            unit.update(status="BLOCKED_UPSTREAM_FAILURE", blocked_by=parent["key"],
                        error="Upstream validation/CAL failed; independent methods continue")
        elif parent["status"] == "COMPLETE_VERIFIED":
            unit["status"] = ready
            if unit["phase"] == "final":
                unit.update(release=parent["release"], selection=parent["selection"])


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
                  "units": matrix(contract, m["family_indices"]), "status": "INDEPENDENT_METHOD_GATES"}
        write(B/"LEDGER.json", ledger)
        for _ in range(94*120):
            if not active():
                write(B/"STATUS.json", {"utc": utc(), "status": "SUPERSEDED_NO_NEW_SUBMISSIONS"}); return
            try:
                rows = queue()
            except (RuntimeError, OSError, subprocess.SubprocessError) as error:
                write(B/"STATUS.json", {"utc": utc(), "status": "QUEUE_RETRY_NO_SUBMISSION", "error": repr(error)})
                time.sleep(30); continue
            bundle()
            for unit in ledger["units"]:
                observe(unit, rows, m, runner, records, ledger)
            advance_waiters(ledger)
            for unit in ledger["units"]:
                if unit["status"] == "READY_TO_RELEASE":
                    try:
                        proofs, reason = validation_reports(unit["family"], rows, m, old, runner, contract, records, ledger)
                        require(proofs is not None, "Full validation was not actually complete: " + repr(reason))
                        unit.update(release=release_family(unit, proofs, m, old, contract), status="READY")
                        write(B/"LEDGER.json", ledger)
                    except Exception as error:
                        unit.update(status="FAILED_REVIEW_REQUIRED", error=repr(error))
                        write(B/"LEDGER.json", ledger)
            advance_waiters(ledger)
            order = {"gate": 0, "full": 1, "cal": 2, "final": 3}
            for unit in sorted(ledger["units"], key=lambda u: order[u["phase"]]):
                if unit["status"] == "READY":
                    try: free = capacity(queue())
                    except (RuntimeError, OSError, subprocess.SubprocessError): break
                    if not free: break
                    submit(unit, m, ledger)
            done = sum(u["status"] == "COMPLETE_VERIFIED" for u in ledger["units"])
            failed = [u["key"] for u in ledger["units"] if u["status"] == "FAILED_REVIEW_REQUIRED"]
            blocked = [u["key"] for u in ledger["units"] if u["status"] == "BLOCKED_UPSTREAM_FAILURE"]
            ledger["status"] = "ACTIVE_WITH_FAILURES_REQUIRING_REVIEW" if failed else "ACTIVE"
            write(B/"LEDGER.json", ledger)
            write(B/"STATUS.json", {"utc": utc(), "status": ledger["status"], "controller": ledger["controller"],
                  "all_jobs_complete": done, "all_jobs_planned": len(ledger["units"]),
                  "all_jobs_submitted": sum(bool(u.get("job")) for u in ledger["units"]),
                  "gates_complete": sum(u["phase"] == "gate" and u["status"] == "COMPLETE_VERIFIED" for u in ledger["units"]),
                  "full_validations_complete": sum(u["phase"] == "full" and u["status"] == "COMPLETE_VERIFIED" for u in ledger["units"]),
                  "formal_jobs_submitted": sum(u["phase"] in ("cal", "final") and bool(u.get("job")) for u in ledger["units"]),
                  "cal_candidates": sum(len(contract.candidates(contract.FAMILIES[i])) for i in m["family_indices"]), "final_units": sum(len(contract.schedule(contract.FAMILIES[i])["final_seeds"]) for i in m["family_indices"]), "failed": failed, "blocked_dependents": blocked,
                  "method_states": {f: {u["key"]: u["status"] for u in ledger["units"] if u["family"] == f} for f in contract.FAMILIES},
                  "v9_completion_dependency": False, "independent_family_progress": True,
                  "shared_normal_gpu_cap": 8, "pending_target": 8, "normal_job_cap": 29,
                  "cancelled_running_jobs": []})
            if done+len(failed)+len(blocked) == len(ledger["units"]):
                write(B/"FINISHED.json", {**ledger, "status": "PASS" if not failed else "REVIEW_REQUIRED"})
                require(not failed, "Formal failures require review: " + repr(failed)); return
            time.sleep(30)
        raise RuntimeError("Supplement controller reached 94-hour limit")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        failure = {"utc": utc(), "status": "FAIL", "error": repr(error), "traceback": traceback.format_exc()}
        write(B/"ERROR.json", failure)
        write(B/"STATUS.json", failure)
        raise

