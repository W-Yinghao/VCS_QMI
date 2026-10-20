"""TEST_REPEAT134 status for the coordinator's feeder (reads the bundle records only; no TEST data, no Slurm query).

    status.py           one line per stage
    status.py --json    {"version": 4, "activation_sha256", "prepare_ready", "input_lock": {"status", "job"},
                         "units_ready": [...], "units_done": [...], "units_blocked": [...], "units_partial": [...],
                         "dev_noise_samples": int}

v4: prepare may run as soon as v4 is frozen; every unit job (one per manifest unit, on the GPU type of its recorded
DEV32) needs INPUT_LOCK PASS under the v4 activation, re-evaluates DEV32, runs the in-job check and reads TEST only if it
passes (else BLOCKED).  Uses the same functions as the jobs (tr_common.records / input_lock).  Run it from the v4
execution tree (or the development worktree at the same records).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_common as C  # noqa: E402


def status() -> dict:
    rec = C.records()
    lock_path = C.BUNDLE / "INPUT_LOCK.json"
    try:
        lock = {"status": "PASS", "job": C.input_lock(rec["activation_sha256"])["job"]}
    except RuntimeError:
        lk = C.read(lock_path) if lock_path.exists() else {}
        lock = {"status": lk.get("status", "absent"), "job": lk.get("job")}
    done, blocked, partial, ready = [], [], [], []
    for u in rec["manifest"]["units"]:
        out = C.BUNDLE / "results/TEST_REPEAT" / u["key"]
        if (out / "COMPLETE.json").exists():
            done.append(u["key"])
        elif (out / "BLOCKED.json").exists():
            blocked.append(u["key"])
        else:
            if out.exists():
                partial.append(u["key"])               # started before (resumable); still ready to (re)submit
            if lock["status"] == "PASS":
                ready.append(u["key"])
    return {"version": 4, "activation_sha256": rec["activation_sha256"], "prepare_ready": not lock_path.exists(),
            "input_lock": lock, "units_ready": ready, "units_done": done, "units_blocked": blocked,
            "units_partial": partial, "dev_noise_samples": len(list(C.BUNDLE.glob("gate/*/*/GATE.json"))) + len(done) + len(blocked)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    st = status()
    if a.json:
        print(json.dumps(st, indent=1))
        return 0
    print(f"v4 prepare_ready={st['prepare_ready']} input_lock={st['input_lock']['status']} ready={len(st['units_ready'])} "
          f"done={len(st['units_done'])}/42 blocked={len(st['units_blocked'])} partial={len(st['units_partial'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
