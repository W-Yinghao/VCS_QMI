"""D2 post-DEV sensitivity: named variant MIND_2D_warp_then_describe (pre-registered).

Slurm L40S:      d2_mind_wtd.py             frozen-variant CAL replay (QC), variant CAL8 x 10, variant DEV32 x 10
Slurm cpu-high:  d2_mind_wtd.py --dry-run   CPU smoke: one CAL core for both readouts, scratch output, no DEV

The frozen MIND_2D_adapted unit (fixed training-free checkpoint), DataGuard, inputs,
positions, searcher, evaluate() and records are reused unchanged from the verified
pipeline tree; only the moving-side MIND energy is the variant (v10_sensitivity.mind_wtd).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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

FAMILY = "MIND_2D_adapted"
PHASE = "D2_POST_DEV_SENSITIVITY"


@contextmanager
def session_class(runner, replacement):
    """Swap the frozen runner's RegistrationSession for the variant subclass, restoring it afterwards."""
    original = runner.RegistrationSession
    C.require(issubclass(replacement, original) and replacement is not original, "Variant session must subclass the frozen one")
    runner.RegistrationSession = replacement
    try:
        yield
    finally:
        runner.RegistrationSession = original


def evaluate_variant_dev(R, records, variant, trainer, guard, data_lock, variant_identity, checkpoint, allocation,
                         cal, out):
    """Variant DEV32 x 10 (the runner's session class must already be the variant), DEV24, per core, strata."""
    C.require(R.RegistrationSession.__name__ == "WarpThenDescribeSession", "DEV must use the variant session")
    reads_before_dev = len(guard.reads)
    R.check_data(data_lock, guard, {"DEV"})
    dev = guard.included("DEV")
    C.require(len(dev) == 32, "Full DEV32 required")
    summary = R.evaluate(trainer, guard, "DEV", dev, out / "VARIANT" / "DEV.jsonl", variant_identity, checkpoint, allocation)
    rows = records.rows(out / "VARIANT" / "DEV.jsonl")
    C.require(len(rows) == 320 and {r["core"] for r in rows} == set(dev), "Incomplete DEV32 coverage")
    C.require(all(r.get("variant") == variant for r in rows), "DEV rows lack the variant label")
    C.require(all(i >= reads_before_dev for i, r in enumerate(guard.reads) if r["role"] == "DEV"),
              "DEV read order violated")
    sensitive = C.dev24_rows(rows, guard.included("FIT"), cal)
    return {"DEV32": summary["metrics"], "DEV24": records.metrics(sensitive),
            "per_core": {c: records.metrics([r for r in rows if r["core"] == c]) for c in dev},
            "dev_strata": S.rotation_strata(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    job = os.environ.get("SLURM_JOB_ID")
    C.require(bool(job), "D2 runs only inside Slurm")
    pre = C.prereg()
    if args.dry_run:
        freeze_sha, code_digest = None, "DRY_RUN"
    else:
        freeze = C.verify_freeze()
        freeze_sha, code_digest = C.sha(C.BUNDLE / "FREEZE.json"), freeze["diagnostic_sources"]["code_sha256"]
        C.require_partition("D2")
    design = pre["D2"]
    ref = pre["frozen_inputs"]["mind_release"]
    release = C.verify_release(ref)
    source_root = Path(ref["source_root"])
    R = Z.initialize(Z.import_runner(source_root), source_root)
    records = sys.modules["v10.supplement.records"]
    from v10_sensitivity import mind_wtd as W
    C.require(W.A is R.A and W.RegistrationSession is R.RegistrationSession
              and Path(sys.modules["v10.supplement.mind"].__file__).resolve().is_relative_to(source_root),
              "Variant is not bound to the frozen implementation")
    C.require(W.VARIANT == design["variant"], "Variant name differs from the pre-registration")
    A, CP = R.A, R.CP

    unit_dir = Path(ref["path"]).parent / ref["unit"]["dir"]
    identity = C.read(unit_dir / "IDENTITY.json")
    C.require(C.sha(unit_dir / "COMPLETE.json") == ref["unit"]["complete_sha256"], "Frozen MIND COMPLETE changed")
    C.require(identity["family"] == FAMILY and identity["seed"] == 1001 and identity["phase"] == "FINAL_DEV"
              and identity["release_sha256"] == ref["sha256"], "Frozen MIND identity differs")
    frozen_complete = records.verified_complete(unit_dir, identity)
    C.require(frozen_complete is not None and frozen_complete["selected"]["step"] == 0, "Frozen MIND unit incomplete")

    device = "cpu" if args.dry_run else "cuda"
    environment = A.environment(None) if args.dry_run else R.runtime_environment()
    data_ref = release["data_lock"]
    C.require(C.sha(data_ref["path"]) == data_ref["sha256"] == pre["frozen_inputs"]["data_lock"]["sha256"],
              "Data lock changed")
    data_lock = C.read(data_ref["path"])
    guard = A.DataGuard()
    R.check_data(data_lock, guard, {"CAL"})
    cal = guard.included("CAL")
    C.require(len(cal) == 8, "Full CAL8 required")
    trainer = R.new_trainer(FAMILY, identity["candidate"], identity["seed"], guard, device)
    C.require(CP.canonical_config(trainer) == C.read(unit_dir / "CONFIG.json"), "Resolved MIND configuration differs")
    checkpoint = CP.load(trainer, unit_dir / "checkpoints" / "step_0000.pt", identity)
    C.require(checkpoint["sha256"] == frozen_complete["artifacts"]["checkpoints/step_0000.pt"], "Fixed checkpoint differs")

    out = (C.OUT_ROOT / "dryrun" / job / "D2") if args.dry_run else (C.OUT_ROOT / "D2" / W.VARIANT)
    base = {"schema": "v10.sensitivity.d2.unit.v1", "diagnostic": design["name"], "label": C.LABEL, "family": FAMILY,
            "candidate": identity["candidate"], "frozen_unit": {"dir": str(unit_dir), "identity": identity,
                                                               "complete_sha256": ref["unit"]["complete_sha256"]},
            "frozen_release_sha256": ref["sha256"], "preregistration_sha256": C.sha(C.PREREGISTRATION),
            "diagnostic_code_sha256": code_digest, "dry_run": args.dry_run}
    replay_identity = {**base, "readout": "frozen MIND_2D_adapted session, CAL replay (QC)"}
    variant_identity = {**base, "readout": "variant", "variant": W.VARIANT, "definition": design["definition"]}
    cal_cores = cal[:1] if args.dry_run else cal
    with records.lock(out / "RUN.lock"):
        if (out / "COMPLETE.json").exists():
            done = C.read(out / "COMPLETE.json")
            C.require(done["identity"] == variant_identity and done["status"] == "COMPLETE", "Stale D2 COMPLETE")
            for name, expected in done["artifacts"].items():
                C.require(C.sha(out / name) == expected, "D2 artifact changed: " + name)
            print("D2_ALREADY_COMPLETE", flush=True)
            return
        C.write_once(out / "IDENTITY.json", variant_identity)
        C.write(out / f"ENV_{job}.json", environment)
        allocation = records.Allocation(out, FAMILY, PHASE)
        status = "FAIL"
        try:
            replay_summary = R.evaluate(trainer, guard, "CAL", cal_cores, out / "REPLAY_FROZEN" / "CAL_step_0000.jsonl",
                                        replay_identity, checkpoint, allocation)
            frozen_cal = [r for r in records.rows(unit_dir / "CAL" / "step_0000.jsonl") if r["core"] in cal_cores]
            replay = S.replay_compare(records.rows(out / "REPLAY_FROZEN" / "CAL_step_0000.jsonl"), frozen_cal)
            C.write(out / "QC_REPLAY_FROZEN_CAL.json", {**replay, "label": C.LABEL,
                                                         "note": "CPU dry run: CPU vs L40S rows" if args.dry_run else
                                                         "L40S replay of the frozen MIND CAL rows"})
            print(f"[D2] frozen-variant CAL replay {replay['status']} ({replay['identical_rows']}/{replay['rows']})", flush=True)
            with session_class(R, W.WarpThenDescribeSession):
                variant_cal = R.evaluate(trainer, guard, "CAL", cal_cores, out / "VARIANT" / "CAL_step_0000.jsonl",
                                         variant_identity, checkpoint, allocation)
                print(f"[D2] variant CAL {variant_cal['metrics']}", flush=True)
                extra = {}
                if not args.dry_run:
                    extra = evaluate_variant_dev(R, records, W.VARIANT, trainer, guard, data_lock, variant_identity,
                                                 checkpoint, allocation, cal, out)
            C.require({r["role"] for r in guard.reads} <= {"CAL", "DEV"}, "Unexpected data role for a training-free method")
            C.write(out / "DATA_READS.json", guard.reads)
            result = {"status": "COMPLETE", "label": C.LABEL, "identity": variant_identity,
                      "replay_frozen_cal": {"metrics": replay_summary["metrics"], "status": replay["status"]},
                      "CAL": variant_cal["metrics"], **extra, "freeze_sha256": freeze_sha, "utc": C.utc(), "job": job}
            result["artifacts"] = {str(p.relative_to(out)): C.sha(p) for p in sorted(out.rglob("*"))
                                   if p.is_file() and p.name not in ("RUN.lock", "COMPLETE.json", "COST_LEDGER.jsonl")
                                   and not p.name.startswith(".")}
            C.write(out / "COMPLETE.json", result)
            status = "COMPLETE"
            print("D2_COMPLETE", flush=True)
        except BaseException as error:
            C.write(out / "failures" / f"{job}.json", {"utc": C.utc(), "error": repr(error),
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
