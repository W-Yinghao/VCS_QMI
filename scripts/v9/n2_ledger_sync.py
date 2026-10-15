"""Fold the V9-N2 release ledger (reports/V9/V9_N2_units/launch_ledger.jsonl) into reports/V9/V9_JOB_IDS.json — one entry per submitted job
with its case (tuning / selected / x1_contingency / contingency_rerun / resubmit:<case>), runs, members and the seed-917 selection snapshot.
Idempotent (keyed by job id).  Refuses to touch V9_JOB_IDS.json when it has uncommitted changes (another agent's edit in progress).

    python scripts/v9/n2_ledger_sync.py            # prints the number of entries added"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "reports/V9/V9_N2_units/launch_ledger.jsonl"
IDS = REPO / "reports/V9/V9_JOB_IDS.json"


def main() -> int:
    if subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "--", str(IDS)]).returncode != 0:
        print("V9_JOB_IDS.json has uncommitted changes — not touching it", file=sys.stderr)
        return 3
    d = json.loads(IDS.read_text())
    known = {j for e in d for j in e.get("jobs", [])}
    added = 0
    for ln in LEDGER.read_text().splitlines():
        ev = json.loads(ln)
        if ev.get("event") != "submitted" or ev["job"] in known:
            continue
        runs = "; ".join(f"rho {r['rho']:g} s{r['seed']} {r['method']} x{r['lr_mult']:g}" + (f" (members {'+'.join(r['members'])})" if len(r["members"]) > 1 else "")
                         for r in ev["runs"])
        entry = {"submitted_utc": ev["submitted_utc"], "unit": f"V9-N2 {ev['kind']} {ev['name']} [{ev['case']}]: {runs}", "commit": ev.get("commit", ""),
                 "job_names": [ev["name"]], "jobs": [ev["job"]], "case": ev["case"], "group_file": str(Path(ev["group_file"]).relative_to(REPO))
                 if ev["group_file"].startswith(str(REPO)) else ev["group_file"], "runs": ev["runs"],
                 "note": ev.get("rule", "") + (f"; resubmission of {ev['resubmit_of']}" if ev.get("resubmit_of") else "")}
        if ev.get("selection"):
            entry["selection_at_submission"] = ev["selection"]
        d.append(entry)
        known.add(ev["job"])
        added += 1
    IDS.write_text(json.dumps(d, indent=1))
    print(f"added {added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
