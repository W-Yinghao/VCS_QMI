"""Write the frozen scope record of the TEST_REPEAT134 input diagnostic (deterministic; --check compares, writes nothing).

    build_diag_authorization.py [--check]

The record changes no v4 rule: INPUT_LOCK stays FAIL and the v4 unit jobs keep refusing.  It names the one core of the
failed lock, the numbers the job may compute, the code it pins and the two preparation windows the cache mtimes are
checked against (V9-G1's prepare log; the v4 prepare job's exit receipt).
"""
from __future__ import annotations

import argparse
import calendar
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_common as C  # noqa: E402

NAME = "DIAGNOSTIC_AUTHORIZATION_V4D.json"
EXEC_ROOT = C.CS / "ssl_pilot_v10_test_repeat_exec_v4diag_20261020"
CODE = ("scripts/v10_test_repeat/diagnose_inputs.py", "slurm/v10_test_repeat/diag.sbatch",
        "scripts/v10_test_repeat/tr_common.py", "scripts/v10_test_repeat/prepare_inputs.py")
V9_PREPARE_LOG = C.CS / "slurm_logs" / "v9_g1test_prepare_1042418.out"


def _unix(stamp: str) -> int:
    return calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))


def v9_window() -> dict:
    text = V9_PREPARE_LOG.read_text()
    head = re.search(r"host=(\S+)\s.*?start=(\S+) commit=(\S+) code=(\S+)", text)
    tail = re.search(r"prepare_rc=(\d+) end=(\S+)", text)
    C.require(head and tail and tail.group(1) == "0", "V9-G1 prepare log is not a completed prepare")
    return {"job": "1042418", "node": head.group(1), "commit": head.group(3), "code_hash": head.group(4),
            "start_utc": head.group(2), "end_utc": tail.group(2), "start_unix": _unix(head.group(2)),
            "end_unix": _unix(tail.group(2)), "source": str(V9_PREPARE_LOG), "source_sha256": C.sha(V9_PREPARE_LOG)}


def v10_window(job: str) -> dict:
    p = C.BUNDLE / "exits" / f"{job}.json"
    r = C.read(p)
    C.require(r["stage"] == "prepare" and r["exit_code"] == 3, "the v4 prepare receipt is not the failed-lock exit")
    return {"job": job, "node": r["nodelist"], "partition": r["partition"],
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r["start_unix"])),
            "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r["end_unix"])),
            "start_unix": int(r["start_unix"]), "end_unix": int(r["end_unix"]) + 1, "source": str(p), "source_sha256": C.sha(p)}


def record() -> dict:
    act_sha = C.sha(C.BUNDLE / C.V4_ACTIVATION)                # the diagnostic sits under the v4 activation
    lock = C.read(C.BUNDLE / "INPUT_LOCK.json")
    C.require(lock["status"] == "FAIL_V9_CACHE_MISMATCH" and lock["activation_sha256"] == act_sha, "unexpected lock state")
    return {
        "schema": "v10.test_repeat134.diagnostic_authorization.v4d", "id": "V10-TEST-REPEAT134-INPUT-DIAGNOSTIC-20261020",
        "requested_by": "the coordinating session, 2026-10-20, after the v4 prepare job " + lock["job"] + " wrote INPUT_LOCK "
                        "FAIL_V9_CACHE_MISMATCH (133 of 134 cores identical to V9-G1's prepared arrays)",
        "v4_activation_sha256": act_sha, "failed_input_lock_sha256": C.sha(C.BUNDLE / "INPUT_LOCK.json"),
        "cores": list(lock["v9_cache_comparison"]["differing"]),
        "scope": "numbers only, for the cores listed: per-array and per-channel differences (count, max and mean abs, float32 "
                 "ULP distance, positions) between the V10 rebuild and V9-G1's prepared arrays; the raw members' sha256 against "
                 "the TEST_REPEAT134 manifest, V9-G1's frozen member CSV, the V10 data manifest and both prepare manifests; "
                 "the cache files' provenance (sha256, mtime, ctime, zip member times and CRC, mtimes of all 134 cache files "
                 "against the two prepare windows, whether V9-G1's prepare manifest pins the cache sha256); rebuilds of the "
                 "core with REG-P1's frozen cmd_prepare in fresh processes under numpy's default CPU dispatch (twice) and with "
                 "X86_V4 or all SIMD targets disabled, compared with both prepared files",
        "not_permitted": ["evaluating any checkpoint or method", "computing perturbations or truths",
                          "reading any registration result, label or V9-G1 result", "reporting element values",
                          "changing any record, lock or prepared input", "keeping a rebuilt copy after the job"],
        "v4_status": "unchanged: INPUT_LOCK.json stays FAIL_V9_CACHE_MISMATCH and every v4 unit job refuses at input_lock "
                     "before any DEV or TEST read; v4 pre-registered no exception or fallback for an input mismatch",
        "windows": {"v9_g1_prepare": v9_window(), "v10_prepare": v10_window(lock["job"])},
        "code": {"files": {rel: C.sha(C.REPO / rel) for rel in CODE}, "execution_root": str(EXEC_ROOT)},
        "job": "pinned Slurm CPU job (cpu-high, no GPU): V10_TR_EXPECT_ACTIVATION = the v4 activation sha256, "
               "V10_TR_EXPECT_DIAG = sha256 of this file, V10_TR_EXPECT_COMMIT = the diagnostic execution commit; output "
               "<bundle>/diagnostics/<core>/<job>/DIAG.json, receipt <bundle>/diagnostics/exits/<job>.json"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = C.dumps(record())
    ok = True
    for path in (C.REPO / C.RECORDS / NAME, C.BUNDLE / NAME):
        if a.check:
            same = path.exists() and path.read_text() == text
            ok &= same
            print(path, "identical" if same else "DIFFERS")
        else:
            C.require(not path.exists() or path.read_text() == text, f"{path} exists with different content")
            path.write_text(text)
            print("wrote", path)
    print(NAME, "sha256", C.sha_bytes(text.encode()))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
