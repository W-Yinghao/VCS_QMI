"""Supplemental BF full runs. No execution until new T01/T02/runner release.

One CAL family or one final seed per job. No submission or queue mutation here.
The first four frozen BF1 families keep their original independent source clone.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from v10.supplement.contract import (FAMILIES, PARENT, candidates, digest, read,
                                    require_release, schedule, sha)
from v10.supplement.records import (Allocation, BudgetExceeded, append, lock, metrics,
                                   require_identity, rows, selection_key, utc,
                                   verified_complete, write)


def identity(release_sha, family, candidate, seed, phase, data_sha, dependency=None):
    return {"schema": "v10.supplement.unit.v1", "release_sha256": release_sha,
            "family": family, "candidate": candidate, "seed": seed, "phase": phase,
            "schedule": schedule(family), "data_lock_sha256": data_sha,
            "selection_dependency_sha256": dependency}


def check_data(data_lock, guard, allowed):
    if data_lock["manifest_sha256"] != guard.manifest_sha256:
        raise RuntimeError("Data lock names a different V10 manifest")
    if sha(data_lock["r6_manifest"]["path"]) != data_lock["r6_manifest"]["sha256"]:
        raise RuntimeError("Underlying R6 manifest changed")
    expected = {(core, role) for role in allowed for core in guard.included(role)}
    actual = [(item["core"], item["role"]) for item in data_lock["files"] if item["role"] in allowed]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("Data lock role coverage differs from guarded data")
    for item in data_lock["files"]:
        if item["role"] in allowed:
            guarded = guard.check(item["core"], {item["role"]}, "Supplemental input integrity")
            if guarded.resolve() != Path(item["path"]).resolve() or sha(guarded) != item["sha256"]:
                raise RuntimeError("Locked image cache changed: " + item["core"])


def runtime_environment():
    environment = A.environment("cuda")
    expected = read(PARENT / "PROTOCOL.json")["precision"]
    if environment.get("gpu_name") != "NVIDIA L40S":
        raise RuntimeError("Supplemental BF budget is defined on L40S")
    for key in ("torch", "cuda_runtime", "cudnn", "torch_threads"):
        if environment[key] != expected["environment"][key]:
            raise RuntimeError("Common runtime environment mismatch: " + key)
    if A.precision_settings(environment["precision"]) != expected["settings"]:
        raise RuntimeError("Common precision settings differ")
    if environment["precision"]["env"] != expected["env"]:
        raise RuntimeError("Common precision environment overrides differ")
    return environment


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def evaluate(trainer, guard, role, cores, path, ident, checkpoint, allocation):
    expected = {"unit": ident, "checkpoint_sha256": checkpoint["sha256"], "role": role,
                "cores": cores, "perturbations": list(range(10)), "searcher": "unchanged R6-REG-P2"}
    id_path = path.with_suffix(".identity.json")
    if id_path.exists() and read(id_path) != expected:
        raise RuntimeError("Evaluation namespace belongs to another identity")
    if not id_path.exists():
        write(id_path, expected)
    records = rows(path)
    seen = {}
    for row in records:
        key = row["core"], row["perturbation"]
        if key in seen or row["core"] not in cores or row["perturbation"] not in range(10):
            raise RuntimeError("Duplicate or unexpected evaluation row")
        if row["role"] != role or row["checkpoint_sha256"] != checkpoint["sha256"]:
            raise RuntimeError("Evaluation mixes data roles or checkpoints")
        if not all(math.isfinite(row[k]) for k in ("err_full", "F_hat", "F_true", "theta_hat", "seconds")):
            raise RuntimeError("Non-finite stored evaluation row")
        seen[key] = row
    summary_path = path.with_suffix(".summary.json")
    if summary_path.exists():
        summary = read(summary_path)
        if (summary.get("status") != "COMPLETE" or summary["identity"] != expected or
                len(seen) != len(cores) * 10 or summary["rows_sha256"] != sha(path) or
                summary["metrics"] != metrics(records)):
            raise RuntimeError("Invalid evaluation summary or coverage")
        return summary
    with CP.evaluation_state(trainer):
        session = RegistrationSession(ident["family"], trainer, guard, role, cores,
                                       trainer.device, ident["candidate"])
        for core in cores:
            for perturbation in range(10):
                if (core, perturbation) in seen:
                    continue
                allocation.check()
                energy, prep = session.energy(core, perturbation)
                A.synchronize(trainer.device); started = time.perf_counter()
                record = plain(A.registration_record(energy, trainer.device,
                                                     session.perturbations[core][perturbation]))
                A.synchronize(trainer.device)
                record["seconds"] = time.perf_counter() - started
                if not all(math.isfinite(record[k]) for k in ("err_full", "F_hat", "F_true", "theta_hat", "seconds")):
                    raise RuntimeError("Non-finite registration result")
                record.update(prep)
                record["full_registration_seconds"] = prep["preparation_seconds"] + record["seconds"]
                record.update(core=core, perturbation=perturbation, role=role,
                              checkpoint_sha256=checkpoint["sha256"], job=os.environ["SLURM_JOB_ID"])
                append(path, record); seen[(core, perturbation)] = record
    if len(seen) != len(cores) * 10:
        raise RuntimeError("Incomplete full registration block")
    summary = {"status": "COMPLETE", "identity": expected, "metrics": metrics(list(seen.values())),
               "rows_sha256": sha(path),
               "measured_complete_registration_seconds": sum(r["full_registration_seconds"] for r in seen.values()),
               "cost_scope": "completed registrations including input/descriptor/latent preparation; allocation also records retries and startup"}
    write(summary_path, summary)
    return summary


def run_unit(directory, ident, data_lock, allocation, environment):
    directory.mkdir(parents=True, exist_ok=True)
    with lock(directory / "RUN.lock"):
        previous = verified_complete(directory, ident)
        if previous is not None:
            return previous
        allocation.check()
        require_identity(directory, ident)
        write(directory / f"ENV_{os.environ['SLURM_JOB_ID']}.json", environment)
        guard = A.DataGuard()
        recipe = schedule(ident["family"])
        roles = {"CAL"} if recipe["kind"] == "training_free" else {"FIT", "CAL"}
        check_data(data_lock, guard, roles)
        cal = guard.included("CAL")
        if len(cal) != 8:
            raise RuntimeError("Full CAL8 required")
        trainer = new_trainer(ident["family"], ident["candidate"], ident["seed"], guard, "cuda")
        configuration = CP.canonical_config(trainer)
        config_path = directory / "CONFIG.json"
        if config_path.exists() and read(config_path) != configuration:
            raise RuntimeError("Existing resolved configuration changed")
        write(config_path, configuration)
        cal_checkpoints = []
        def cal_at(step):
            cp_path = directory / "checkpoints" / f"step_{step:04d}.pt"
            cp = CP.load(trainer, cp_path, ident)
            if cp["step"] != step:
                raise RuntimeError("CAL checkpoint filename/step mismatch")
            result = evaluate(trainer, guard, "CAL", cal, directory / "CAL" / f"step_{step:04d}.jsonl",
                               ident, cp, allocation)
            cal_checkpoints.append({"step": step, "metrics": result["metrics"],
                                    "checkpoint": str(cp_path.relative_to(directory)), "checkpoint_sha256": cp["sha256"]})
        try:
            checkpoints = sorted((directory / "checkpoints").glob("step_*.json"))
            if checkpoints:
                last = max(checkpoints, key=lambda p: read(p)["step"])
                latest = CP.load(trainer, last.with_suffix(".pt"), ident)["step"]
                if latest not in recipe["save_steps"]:
                    raise RuntimeError("Unexpected resume checkpoint")
                for step in recipe["save_steps"]:
                    if step <= latest:
                        meta = CP.metadata(directory / "checkpoints" / f"step_{step:04d}.pt", ident)
                        if meta["step"] != step:
                            raise RuntimeError("Checkpoint sequence step mismatch")
                for step in recipe["cal_steps"]:
                    if step <= latest:
                        cal_at(step)
                CP.load(trainer, last.with_suffix(".pt"), ident)
            elif recipe["steps"] == 0:
                CP.save(trainer, directory / "checkpoints/step_0000.pt", ident)
                cal_at(0)
            while trainer.n_steps < recipe["steps"]:
                allocation.check()
                record = plain(trainer.step())
                for key in ("loss", "J", "grad_norm_preclip"):
                    if key in record and not math.isfinite(record[key]):
                        raise RuntimeError("Non-finite training value: " + key)
                record.update(job=os.environ["SLURM_JOB_ID"], utc=utc())
                append(directory / "TRAIN_STEPS.jsonl", record)
                step = trainer.n_steps
                if step in recipe["save_steps"]:
                    CP.save(trainer, directory / "checkpoints" / f"step_{step:04d}.pt", ident)
                    allocation.record("CHECKPOINT", unit=directory.name, step=step)
                if step in recipe["cal_steps"]:
                    cal_at(step)
            if [c["step"] for c in cal_checkpoints] != recipe["cal_steps"]:
                raise RuntimeError("Prescribed full CAL checkpoint set missing")
            chosen = min(cal_checkpoints, key=selection_key)
            extra = {}
            if ident["phase"] == "FINAL_DEV":
                check_data(data_lock, guard, {"DEV"})
                dev = guard.included("DEV")
                if len(dev) != 32:
                    raise RuntimeError("Full DEV32 required")
                cp = CP.load(trainer, directory / chosen["checkpoint"], ident)
                result = evaluate(trainer, guard, "DEV", dev, directory / "DEV.jsonl", ident, cp, allocation)
                records = rows(directory / "DEV.jsonl")
                excluded = {c.split("_", 1)[1] for c in guard.included("FIT") + cal}
                sensitive = [r for r in records if r["core"].split("_", 1)[1] not in excluded]
                if len({r["core"] for r in sensitive}) != 24:
                    raise RuntimeError("Frozen DEV24 sensitivity set changed")
                extra = {"DEV32": result["metrics"], "DEV24": metrics(sensitive),
                         "per_core": {c: metrics([r for r in records if r["core"] == c]) for c in dev}}
            write(directory / "DATA_READS.json", guard.reads)
            result = {"status": "COMPLETE", "identity": ident, "steps": recipe["steps"],
                      "work_kind": recipe["kind"], "cal_checkpoints": cal_checkpoints,
                      "selected": chosen, "utc": utc(), "job": os.environ["SLURM_JOB_ID"], **extra}
            result["artifacts"] = {str(p.relative_to(directory)): sha(p) for p in directory.rglob("*")
                                   if p.is_file() and p.name not in ("RUN.lock", "COMPLETE.json")
                                   and not p.name.startswith(".")}
            write(directory / "COMPLETE.json", result)
            return verified_complete(directory, ident)
        except BaseException:
            write(directory / f"READS_INCOMPLETE_{os.environ['SLURM_JOB_ID']}.json", guard.reads)
            raise


def checked_selection(root, family, release_sha, data_sha, require_exit=False):
    path = root / "CAL" / family / "SELECTION.json"
    selection = read(path)
    if (selection.get("status") != "SELECTED" or selection.get("family") != family or
            selection.get("release_sha256") != release_sha):
        raise RuntimeError("Selection identity mismatch")
    if require_exit:
        receipt = read(root.parent / "exits" / f"{selection['job']}.json")
        if (receipt.get("job") != selection["job"] or receipt.get("exit_code") != 0 or
                receipt.get("release_sha256") != release_sha):
            raise RuntimeError("CAL selection lacks a successful exit for this release")
    complete = selection["completed_candidates"]
    planned = candidates(family)
    if not complete or [r["candidate"] for r in complete] != planned[:len(complete)]:
        raise RuntimeError("CAL candidate sequence is incomplete or reordered")
    if len(complete) != len(planned) and not selection.get("budget_exhausted"):
        raise RuntimeError("Missing full CAL candidates without budget evidence")
    if selection.get("budget_exhausted"):
        budget = read(path.parent / "BUDGET_EXHAUSTED.json")
        if not math.isfinite(budget["allocated_seconds"]) or budget["allocated_seconds"] < 8 * 3600 - 30:
            raise RuntimeError("Budget-limited selection lacks actual cap exhaustion")
    for record in complete:
        unit = root / "CAL" / family / f"candidate_{record['candidate']['order']:02d}"
        ident = identity(release_sha, family, record["candidate"], 1017, "CAL", data_sha)
        result = verified_complete(unit, ident)
        if (result is None or sha(unit / "COMPLETE.json") != record["complete_sha256"] or
                {**result["selected"], "order": record["candidate"]["order"]} != record["choice"]):
            raise RuntimeError("Selection dependency changed")
    if min(complete, key=lambda r: selection_key(r["choice"])) != selection["selected"]:
        raise RuntimeError("CAL selection differs from fixed lexicographic rule")
    return selection, sha(path)


def run_cal(root, family, release_sha, data_sha, data_lock, allocation, environment):
    directory = root / "CAL" / family
    if (directory / "SELECTION.json").exists():
        return checked_selection(root, family, release_sha, data_sha)
    completed, exhausted = [], False
    for candidate in candidates(family):
        unit = directory / f"candidate_{candidate['order']:02d}"
        ident = identity(release_sha, family, candidate, 1017, "CAL", data_sha)
        try:
            result = run_unit(unit, ident, data_lock, allocation, environment)
        except BudgetExceeded:
            exhausted = True
            # Disable the elapsed-time alarm to write terminal selection/accounting.
            signal.setitimer(signal.ITIMER_REAL, 0)
            write(directory / "BUDGET_EXHAUSTED.json", {"utc": utc(), "candidate": candidate,
                  "allocated_seconds": allocation.previous + allocation.elapsed(),
                  "partial_candidate_excluded": True})
            break
        completed.append({"candidate": candidate, "complete_sha256": sha(unit / "COMPLETE.json"),
                          "choice": {**result["selected"], "order": candidate["order"]}})
    if not completed:
        raise RuntimeError("BLOCKED: no complete full-length candidate within CAL budget")
    selection = {"status": "SELECTED", "family": family, "release_sha256": release_sha,
                 "completed_candidates": completed, "budget_exhausted": exhausted,
                 "selected": min(completed, key=lambda r: selection_key(r["choice"])), "utc": utc(),
                 "job": os.environ["SLURM_JOB_ID"]}
    write(directory / "SELECTION.json", selection)
    return checked_selection(root, family, release_sha, data_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("cal", "final"))
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.phase == "final" and args.seed not in schedule(args.family)["final_seeds"]:
        parser.error("final requires one of the method's predefined final seeds")
    if args.phase == "cal" and args.seed is not None:
        parser.error("CAL seed is fixed at 1017")
    # Pure-record validation before importing Torch or touching any images.
    release = require_release(args.release, args.family)
    base = args.release.resolve().parent
    root = base / "results"
    release_sha = sha(args.release)
    data_lock_path = base / release["data_lock"]["path"]
    data_sha, data_lock = sha(data_lock_path), read(data_lock_path)
    global A, torch, np, CP, new_trainer, RegistrationSession
    import numpy as np
    import torch
    from v10 import adapter as A
    from v10.supplement import checkpoints as CP
    from v10.supplement.training import new_trainer
    from v10.supplement.registration import RegistrationSession

    def terminated(signum, frame):
        raise InterruptedError(f"Slurm signal {signum}; preserve last verified checkpoint")
    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGUSR1, terminated)
    # CAL serialization also protects family budget accounting; finals have their
    # own output locks, so the three independent seeds can run concurrently.
    lock_path = (root / "CAL" / args.family / "FAMILY.lock" if args.phase == "cal" else
                 root / "FINAL_DEV" / args.family / f"seed_{args.seed}.lock")
    with lock(lock_path):
        allocation = Allocation(base, args.family, "CAL" if args.phase == "cal" else "FINAL_DEV")
        status = "FAIL"
        try:
            environment = runtime_environment()
            if args.phase == "cal":
                def budget_alarm(signum, frame):
                    raise BudgetExceeded("CAL wall-time alarm reached")
                signal.signal(signal.SIGALRM, budget_alarm)
                remaining = 8 * 3600 - 30 - allocation.previous - allocation.elapsed()
                if remaining > 0:
                    signal.setitimer(signal.ITIMER_REAL, remaining)
                run_cal(root, args.family, release_sha, data_sha, data_lock, allocation, environment)
            else:
                selection, dependency = checked_selection(root, args.family, release_sha, data_sha, require_exit=True)
                candidate = selection["selected"]["candidate"]
                ident = identity(release_sha, args.family, candidate, args.seed, "FINAL_DEV", data_sha, dependency)
                run_unit(root / "FINAL_DEV" / args.family / f"seed_{args.seed}", ident,
                         data_lock, allocation, environment)
            status = "COMPLETE"
        except BaseException as error:
            write(base / "failures" / f"{os.environ['SLURM_JOB_ID']}.json",
                  {"utc": utc(), "family": args.family, "phase": args.phase,
                   "exception": repr(error), "traceback": traceback.format_exc()})
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            allocation.record("END", status=status)


if __name__ == "__main__":
    main()
