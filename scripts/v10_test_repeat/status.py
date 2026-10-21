"""TEST_REPEAT134 status for the coordinator's feeder (reads the bundle records only; no TEST data, no Slurm query).

    status.py           one line per stage
    status.py --json    {"version": 6, "activation_sha256", "v6_units", "input_variant", "input_lock": {"status", "job"},
                         "units_ready": [...], "units_done": [...], "units_blocked": [...], "units_partial": [...],
                         "dev_noise_samples": int}

v6: units_ready lists only the 3 CoMIR_native units of the v6 activation (outputs in results/TEST_REPEAT_V6); units_done /
blocked / partial cover all 42 (the other 39 from their v5 outputs).  v5: the inputs job (cpu.sbatch inputs) ran once; every unit
job (one per manifest unit, on the GPU type of its recorded DEV32) needs INPUT_LOCK_V5 PASS under the v5 activation,
re-evaluates DEV32, runs the in-job check and reads TEST only if it passes (else BLOCKED).  v4's INPUT_LOCK.json stays the
FAIL record it is.  Uses the same functions as the jobs (tr_common.records / input_lock_v5).  Run it from the v5 execution
tree (or the development worktree at the same records).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_common as C  # noqa: E402


def status() -> dict:
    """v6: the 3 CoMIR_native units (results/TEST_REPEAT_V6/<unit>); the other 39 units are v5's and are reported as found."""
    rec = C.records()
    v6 = rec["activation"]["v6"]
    lock_path = C.BUNDLE / C.LOCK
    try:
        lock = {"status": "PASS", "job": C.input_lock_v5(v6["input_lock_activation_sha256"], rec["protocol"])["job"]}
    except RuntimeError:
        lk = C.read(lock_path) if lock_path.exists() else {}
        lock = {"status": lk.get("status", "absent"), "job": lk.get("job")}
    done, blocked, partial, ready = [], [], [], []
    for u in rec["manifest"]["units"]:
        k = u["key"]
        out = C.BUNDLE / (C.RESULTS_V6 if k in v6["units"] else "results/TEST_REPEAT") / k
        if (out / "COMPLETE.json").exists():
            done.append(k)
        elif (out / "BLOCKED.json").exists():
            blocked.append(k)
        else:
            if out.exists():
                partial.append(k)               # started before (resumable); a v6 unit is still ready to (re)submit
            if lock["status"] == "PASS" and k in v6["units"]:
                ready.append(k)
    return {"version": 6, "activation_sha256": rec["activation_sha256"], "v6_units": v6["units"],
            "input_variant": rec["protocol"]["inputs"]["v5"]["variant"], "input_lock": lock, "units_ready": ready,
            "units_done": done, "units_blocked": blocked, "units_partial": partial,
            "dev_noise_samples": len(list(C.BUNDLE.glob("gate/*/*/GATE.json"))) + len(done) + len(blocked)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    st = status()
    if a.json:
        print(json.dumps(st, indent=1))
        return 0
    print(f"v6 units={len(st['v6_units'])} input_lock={st['input_lock']['status']} ready(v6)={len(st['units_ready'])} "
          f"done={len(st['units_done'])}/42 blocked={len(st['units_blocked'])} partial={len(st['units_partial'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
