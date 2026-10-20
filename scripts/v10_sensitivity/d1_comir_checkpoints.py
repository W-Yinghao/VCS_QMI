"""D1 post-DEV sensitivity: CoMIR_native epoch-checkpoint selection on CAL (pre-registered).

Slurm L40S:        d1_comir_checkpoints.py RUN_KEY             RUN_KEY: CAL_1017 FINAL_1001 FINAL_1002 FINAL_1003
Slurm cpu-high:    d1_comir_checkpoints.py RUN_KEY --dry-run   CPU smoke: one CAL core at step 736, scratch output, no DEV

The frozen CoMIR unit, its saved epoch checkpoints and the frozen supplement runner
(scripts/v10_supplement/run.py evaluate, RegistrationSession, checkpoints.load,
records.selection_key) are reused unchanged from the verified native-recovery tree.
Nothing is trained and nothing in the frozen tree or bundle is written.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import traceback

DIAG = Path(__file__).resolve().parents[2]
sys.path.append(str(DIAG / "src"))  # appended: the frozen tree is put first by frozen.import_runner

from v10_sensitivity import analysis as S  # noqa: E402
from v10_sensitivity import common as C  # noqa: E402
from v10_sensitivity import frozen as Z  # noqa: E402

FAMILY = "CoMIR_native"
PHASE = "D1_POST_DEV_SENSITIVITY"


def frozen_unit(ref, key, records):
    unit = ref["units"][key]
    directory = Path(ref["path"]).parent / unit["dir"]
    identity = C.read(directory / "IDENTITY.json")
    C.require(C.sha(directory / "COMPLETE.json") == unit["complete_sha256"], "Frozen COMPLETE changed: " + key)
    C.require(identity["family"] == FAMILY and identity["seed"] == unit["seed"] and identity["phase"] == unit["phase"]
              and identity["release_sha256"] == ref["sha256"], "Frozen unit identity differs: " + key)
    if unit["phase"] == "FINAL_DEV":
        C.require(identity["selection_dependency_sha256"] == ref["cal_selection_sha256"], "Final unit names another CAL selection")
    complete = records.verified_complete(directory, identity)
    C.require(complete is not None and [c["step"] for c in complete["cal_checkpoints"]] == [736]
              and complete["selected"]["step"] == 736, "Frozen CoMIR unit is not the single epoch-23 CAL readout")
    return unit, directory, identity, complete


def evaluate_dev(R, records, trainer, guard, data_lock, unit_dir, identity, d1_identity, chosen, steps, out,
                 allocation, cal):
    """DEV32 x 10 only at the written CAL selection; DEV24 and per-core from those rows; replay if step 736."""
    step = chosen["step"]
    C.require(step in steps and (out / "SELECTION.json").exists()
              and C.read(out / "SELECTION.json")["selected"] == chosen, "DEV requires the written CAL selection")
    checkpoint = R.CP.load(trainer, unit_dir / "checkpoints" / f"step_{step:04d}.pt", identity)
    C.require(checkpoint["step"] == step, "DEV checkpoint is not the selected one")
    R.check_data(data_lock, guard, {"DEV"})
    dev = guard.included("DEV")
    C.require(len(dev) == 32, "Full DEV32 required")
    summary = R.evaluate(trainer, guard, "DEV", dev, out / "DEV.jsonl", d1_identity, checkpoint, allocation)
    rows = records.rows(out / "DEV.jsonl")
    C.require(len(rows) == 320 and {r["core"] for r in rows} == set(dev), "Incomplete DEV32 coverage")
    sensitive = C.dev24_rows(rows, guard.included("FIT"), cal)
    extra = {"DEV_step": step, "DEV_epoch": step // 32, "DEV32": summary["metrics"],
             "DEV24": records.metrics(sensitive),
             "per_core": {c: records.metrics([r for r in rows if r["core"] == c]) for c in dev}}
    if step == 736:
        extra["QC_REPLAY_DEV"] = S.replay_compare(rows, records.rows(unit_dir / "DEV.jsonl"))
        C.write(out / "QC_REPLAY_DEV_0736.json", {**extra["QC_REPLAY_DEV"], "label": C.LABEL})
    return extra


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_key", choices=("CAL_1017", "FINAL_1001", "FINAL_1002", "FINAL_1003"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    job = os.environ.get("SLURM_JOB_ID")
    C.require(bool(job), "D1 runs only inside Slurm")
    pre = C.prereg()
    if args.dry_run:
        freeze_sha, code_digest = None, "DRY_RUN"
    else:
        freeze = C.verify_freeze()
        freeze_sha, code_digest = C.sha(C.BUNDLE / "FREEZE.json"), freeze["diagnostic_sources"]["code_sha256"]
        C.require(freeze["preregistration"]["sha256"] == C.sha(C.PREREGISTRATION), "Pre-registration differs")
        C.require_partition("D1")
    design = pre["D1"]
    steps = design["steps"]
    C.require(steps == [32 * e for e in design["epochs"]] and len(steps) == 12 and steps[-1] == 736,
              "Pre-registered D1 checkpoint set changed")
    ref = pre["frozen_inputs"]["comir_release"]
    release = C.verify_release(ref)
    source_root = Path(ref["source_root"])
    R = Z.initialize(Z.import_runner(source_root), source_root)
    records = sys.modules["v10.supplement.records"]
    C.require(Path(records.__file__).resolve().is_relative_to(source_root), "records module not frozen")
    A, CP = R.A, R.CP
    unit, unit_dir, identity, frozen_complete = frozen_unit(ref, args.run_key, records)
    for step in steps:
        C.require(f"checkpoints/step_{step:04d}.pt" in frozen_complete["artifacts"], f"Frozen checkpoint {step} missing")

    device = "cpu" if args.dry_run else "cuda"
    environment = A.environment(None) if args.dry_run else R.runtime_environment()
    data_ref = release["data_lock"]
    C.require(C.sha(data_ref["path"]) == data_ref["sha256"] == pre["frozen_inputs"]["data_lock"]["sha256"],
              "Data lock changed")
    data_lock = C.read(data_ref["path"])
    guard = A.DataGuard()
    R.check_data(data_lock, guard, {"FIT", "CAL"})
    cal = guard.included("CAL")
    C.require(len(cal) == 8, "Full CAL8 required")
    trainer = R.new_trainer(FAMILY, identity["candidate"], identity["seed"], guard, device)
    C.require(CP.canonical_config(trainer) == C.read(unit_dir / "CONFIG.json"), "Resolved CoMIR configuration differs")

    out = (C.OUT_ROOT / "dryrun" / job / "D1" / args.run_key) if args.dry_run else (C.OUT_ROOT / "D1" / args.run_key)
    d1_identity = {"schema": "v10.sensitivity.d1.unit.v1", "diagnostic": design["name"], "label": C.LABEL,
                   "family": FAMILY, "candidate": identity["candidate"], "run_key": args.run_key,
                   "frozen_unit": {"dir": str(unit_dir), "identity": identity, "complete_sha256": unit["complete_sha256"]},
                   "frozen_release_sha256": ref["sha256"], "preregistration_sha256": C.sha(C.PREREGISTRATION),
                   "steps": steps, "selection_rule": pre["common"]["selection_rule"],
                   "diagnostic_code_sha256": code_digest, "dry_run": args.dry_run}
    cal_cores = cal[:1] if args.dry_run else cal
    evaluated = [736] if args.dry_run else steps
    with records.lock(out / "RUN.lock"):
        if (out / "COMPLETE.json").exists():
            done = C.read(out / "COMPLETE.json")
            C.require(done["identity"] == d1_identity and done["status"] == "COMPLETE", "Stale D1 COMPLETE")
            for name, expected in done["artifacts"].items():
                C.require(C.sha(out / name) == expected, "D1 artifact changed: " + name)
            print("D1_ALREADY_COMPLETE", args.run_key, flush=True)
            return
        C.write_once(out / "IDENTITY.json", d1_identity)
        C.write(out / f"ENV_{job}.json", environment)
        allocation = records.Allocation(out, FAMILY, PHASE)
        status = "FAIL"
        try:
            cal_rows = []
            for step in evaluated:
                checkpoint_path = unit_dir / "checkpoints" / f"step_{step:04d}.pt"
                checkpoint = CP.load(trainer, checkpoint_path, identity)
                C.require(checkpoint["step"] == step and checkpoint["sha256"] ==
                          frozen_complete["artifacts"][f"checkpoints/step_{step:04d}.pt"], "Checkpoint identity mismatch")
                summary = R.evaluate(trainer, guard, "CAL", cal_cores, out / "CAL" / f"step_{step:04d}.jsonl",
                                     d1_identity, checkpoint, allocation)
                C.require(summary["metrics"]["n"] == 10 * len(cal_cores), "Incomplete CAL readout")
                cal_rows.append({"step": step, "epoch": step // 32, "metrics": summary["metrics"],
                                 "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint["sha256"],
                                 "model_sha256": checkpoint["model_sha256"],
                                 "measured_complete_registration_seconds": summary["measured_complete_registration_seconds"]})
                print(f"[D1 {args.run_key}] epoch {step // 32:2d} CAL success<42 {summary['metrics']['success42']:.4f} "
                      f"<9 {summary['metrics']['success9']:.4f} median {summary['metrics']['median_err']:.3f}", flush=True)
            chosen = min(cal_rows, key=records.selection_key)
            selection = {"status": "SELECTED", "label": C.LABEL, "rule": pre["common"]["selection_rule"],
                         "rule_implementation": "frozen records.selection_key (items 1-4; one candidate)",
                         "rows": cal_rows, "selected": chosen, "dry_run": args.dry_run}
            if (out / "SELECTION.json").exists():
                previous = C.read(out / "SELECTION.json")
                C.require(previous["rows"] == cal_rows and previous["selected"] == chosen, "Existing D1 selection differs")
            else:
                C.write(out / "SELECTION.json", {**selection, "utc": C.utc(), "job": job})
            reads_before_dev = len(guard.reads)
            frozen_cal = [r for r in records.rows(unit_dir / "CAL" / "step_0736.jsonl") if r["core"] in cal_cores]
            replay_cal = S.replay_compare(records.rows(out / "CAL" / "step_0736.jsonl"), frozen_cal)
            C.write(out / "QC_REPLAY_CAL_0736.json", {**replay_cal, "label": C.LABEL,
                                                       "note": "CPU dry run: CPU vs L40S rows" if args.dry_run else
                                                       "L40S replay of the frozen epoch-23 CAL rows"})
            extra = {}
            if unit["phase"] == "FINAL_DEV" and not args.dry_run:
                extra = evaluate_dev(R, records, trainer, guard, data_lock, unit_dir, identity, d1_identity, chosen,
                                     steps, out, allocation, cal)
            dev_reads = [i for i, r in enumerate(guard.reads) if r["role"] == "DEV"]
            C.require(all(i >= reads_before_dev for i in dev_reads), "DEV was read before the CAL selection")
            C.require({r["role"] for r in guard.reads} <= {"FIT", "CAL", "DEV"}, "Unexpected data role")
            C.write(out / "DATA_READS.json", guard.reads)
            result = {"status": "COMPLETE", "label": C.LABEL, "identity": d1_identity, "selected": chosen,
                      "cal_rows": cal_rows, "qc_replay_cal_0736": replay_cal["status"], **extra,
                      "freeze_sha256": freeze_sha, "utc": C.utc(), "job": job}
            result["artifacts"] = {str(p.relative_to(out)): C.sha(p) for p in sorted(out.rglob("*"))
                                   if p.is_file() and p.name not in ("RUN.lock", "COMPLETE.json", "COST_LEDGER.jsonl")
                                   and not p.name.startswith(".")}
            C.write(out / "COMPLETE.json", result)
            status = "COMPLETE"
            print("D1_COMPLETE", args.run_key, "selected epoch", chosen["epoch"], flush=True)
        except BaseException as error:
            C.write(out / "failures" / f"{job}.json", {"utc": C.utc(), "run_key": args.run_key, "error": repr(error),
                                                      "traceback": traceback.format_exc(), "label": C.LABEL})
            C.write(out / f"READS_INCOMPLETE_{job}.json", guard.reads)
            raise
        finally:
            allocation.record("END", status=status)


def interrupted(signum, frame):
    raise InterruptedError(f"Slurm signal {signum}; completed rows are kept for an identity-checked resume")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGUSR1, interrupted)
    main()
