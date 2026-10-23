"""Execute one row of an r22 JOBS csv inside an allocated job; no scheduler calls.

The engine directory next to this file is put first on sys.path. The held-out guard (guard.py) is installed before the
trainer starts. KD: after a complete run, last.pt is deleted (only best.pt is kept; the trainer has no resume path
used here). AIG: the trainer itself saves only best.pt.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", required=True)
    p.add_argument("--job-name", required=True)
    p.add_argument("--grad-check", action="store_true", help="run grad_check.py on this row's config first (smokes)")
    a = p.parse_args()
    matrix = Path(a.matrix).resolve()
    rows = [r for r in csv.DictReader(matrix.open(newline="")) if r["job_name"] == a.job_name]
    if len(rows) != 1:
        raise SystemExit(f"job name {a.job_name} is not unique in {matrix}")
    row = rows[0]
    config = matrix.parent / row["config"]
    if hashlib.sha256(config.read_bytes()).hexdigest() != row["config_sha256"]:
        raise SystemExit("config differs from the JOBS csv")
    cfg = json.loads(config.read_text())
    engine = HERE / "engine"
    sys.path[:0] = [str(HERE), str(engine), str(engine / "shared")]
    import guard
    guard.install(guard.kd_roots(cfg) if row["app"] == "kd" else guard.aig_roots(cfg))
    if a.grad_check:
        import grad_check
        out = Path(row["run_dir"]).parent / (Path(row["run_dir"]).name + "__grad_check.json")
        grad_check.run(row["app"], cfg, config, out)
    start = time.time()
    if row["app"] == "kd":
        from kd.runner import train
        result = train(config, row["run_dir"])
        last = Path(row["run_dir"]) / "last.pt"
        if result["status"] == "complete" and last.exists():
            last.unlink()
    elif row["app"] == "aigiqa":
        from aigiqa.engine import train
        if cfg["output_dir"] != row["run_dir"]:
            raise SystemExit("matrix and config output disagree")
        result = train(cfg)
    else:
        raise SystemExit("unsupported app")
    print(json.dumps({"job_name": a.job_name, "status": result["status"], "seconds": time.time() - start,
                      "guard": guard.status(), "test_evaluated": False}))


if __name__ == "__main__":
    main()
