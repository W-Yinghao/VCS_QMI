"""AIGIQA-relevant commands of the package's scripts/run_validation.py, each in its own interpreter, same pass rules.

Runs all six even if one fails (the package runner stops at the first failure, and its KD steps precede aigiqa_tests).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.pkg).resolve(), Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "shared"), env.get("PYTHONPATH", "")])
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"  # as the package runner
    commands = [("dependence_tests", ["-m", "unittest", "discover", "-s", "shared/tests", "-p", "test_dependence.py", "-v"]),
                ("experiment_tests", ["-m", "unittest", "discover", "-s", "shared/tests", "-p", "test_experiment.py", "-v"]),
                ("math", ["-m", "vcs_common.probes", "math", "--output", str(out / "math.json")]),
                ("variance", ["-m", "vcs_common.probes", "variance", "--output", str(out / "sampling.json")]),
                ("aigiqa_tests", ["-m", "unittest", "discover", "-s", "aigiqa/tests", "-v"]),
                ("aigiqa_smoke", ["-m", "aigiqa.cli", "smoke", "--output", str(out / "aigiqa_smoke")])]
    rows = []
    for name, cmd in commands:
        log = out / f"{name}.log"
        t0 = time.perf_counter()
        with log.open("w") as stream:
            rc = subprocess.run([sys.executable] + cmd, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        text = log.read_text()
        row = {"name": name, "command": [sys.executable] + cmd, "returncode": rc, "seconds": time.perf_counter() - t0,
               "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(), "passed": rc == 0}
        if name.endswith("_tests"):
            m = re.search(r"Ran (\d+) tests? in ", text)
            row["tests_completed"] = int(m.group(1)) if m else 0
            row["passed"] = row["passed"] and row["tests_completed"] > 0 and bool(re.search(r"^OK\s*$", text, re.MULTILINE))
        rows.append(row)
        print(name, "PASS" if row["passed"] else "FAIL", flush=True)
    summary = {"status": "PASS" if all(r["passed"] for r in rows) else "FAIL", "scope": "CPU synthetic fixtures (package runner subset)",
               "real_ImageReward_pretrained_tested": False, "real_dataset_training_tested": False, "cuda_tested": False, "commands": rows}
    (out / "SUBSET_SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    sys.exit(0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
