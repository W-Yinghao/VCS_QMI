"""Map a pilot config stem (e.g. vcs_exact_text_w0.1_s3101) to its registered job_id in the global lock.

Read-only. Verifies the lock's own hash and refuses (exit 5) when the job already has a receipt, so a duplicate
submission never reaches `vcs_common.experiment train` (which would refuse it too, after hashing every asset).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vcs_common.experiment import canonical_hash, _receipt_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    lock = json.loads(Path(a.lock).read_text(encoding="utf-8"))
    body = dict(lock); stored = body.pop("lock_sha256", None)
    if canonical_hash(body) != stored:
        sys.exit("lock hash mismatch")
    jobs = [j for j in lock["jobs"] if Path(j["config_path"]).stem == a.name]
    if len(jobs) != 1:
        sys.exit(f"expected exactly one registered job for {a.name}, found {len(jobs)}")
    receipt = _receipt_path(a.lock, jobs[0])
    if receipt.exists():
        status = json.loads(receipt.read_text()).get("status")
        print(f"job {jobs[0]['job_id']} already attempted (receipt status {status}); not resubmittable under this lock", file=sys.stderr)
        sys.exit(5)
    print(jobs[0]["job_id"])


if __name__ == "__main__":
    main()
