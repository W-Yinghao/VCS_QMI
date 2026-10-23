#!/usr/bin/env python3
"""GPU-hour ledger of the V10 app round (cap 21 GPU h over all attempts: smokes, pairs, diagnostics, failures).

Used = per-job receipt (Slurm StartTime -> end of the wrapper).  A submitted GPU job without a receipt is counted at its full
--time limit (conservative) until it has one.  `ledger.py --add HH:MM:SS` exits 1 when used + outstanding + add > cap."""
import json
import sys
from pathlib import Path

L = Path("/home/infres/yinwang/CS_QMI/handoff/v10_app_round_20261021/ledger")
CAP = 21.0


def hours(limit):
    h, m, s = map(int, limit.split(":"))
    return h + m / 60 + s / 3600


rows = [l.split("\t") for l in (L / "SUBMITTED.tsv").read_text().splitlines() if l.strip()] if (L / "SUBMITTED.tsv").exists() else []
used = outstanding = 0.0
jobs = []
for job, name, limit, kind, utc, *_ in rows:
    if kind != "gpu":
        continue
    receipt = L / "receipts" / f"{job}.json"
    if receipt.exists():
        r = json.loads(receipt.read_text())
        h = r["allocated_seconds"] / 3600
        used += h
        jobs.append({"job": job, "name": name, "state": f"receipt exit={r['exit_code']}", "gpu": r.get("gpu"), "hours": round(h, 4)})
    else:
        outstanding += hours(limit)
        jobs.append({"job": job, "name": name, "state": "no receipt: --time counted", "hours": hours(limit)})
add = hours(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--add" else 0.0
report = {"cap_gpu_hours": CAP, "used_gpu_hours": round(used, 4), "outstanding_limit_hours": round(outstanding, 4),
          "committed_gpu_hours": round(used + outstanding, 4), "requested_add": add, "jobs": jobs}
print(json.dumps(report, indent=1))
sys.exit(1 if used + outstanding + add > CAP + 1e-9 else 0)
