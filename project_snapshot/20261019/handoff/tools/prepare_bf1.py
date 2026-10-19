#!/usr/bin/env python3
"""Materialize the authorized BF1 candidate queue and current readiness evidence.

No new scientific choices, protocol freeze, cluster submission or held-out data
access. The queue is NOT executable until its driver and freeze gates are done.
"""
from __future__ import annotations
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from audit_v10 import tracked_hash

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "handoff"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    pp = ROOT / "ssl_pilot_v10/v10/protocol/V10_PROTOCOL_P2_20261017.json"
    bp = ROOT / "ssl_pilot_v10/v10/T02/BUDGET.json"
    dp = ROOT / "V10_PROTOCOL_DECISIONS_20261017.json"
    p, b, d = read(pp), read(bp), read(dp)
    audit = read(OUT / "evidence/V10_AUDIT.json")
    if audit["status"] != "PASS":
        raise SystemExit("V10 audit must pass before preparing the BF1 queue")
    cs = p["bf_batch1"]["candidates"]
    prices = {r["candidate_id"]: r for r in b["projection"]["candidates"]}
    assert len(cs) == len(prices) == 18 and set(prices) == {c["candidate_id"] for c in cs}
    assert p["bf_batch1"]["methods"] == d["rollout"]["first_natural_families"]
    queue = []
    for c in cs:
        cost = prices[c["candidate_id"]]
        assert cost["lr"] == c["lr"] and cost["m1_cap"] == c["m1_cap"]
        queue.append({"unit_id": c["candidate_id"], "block": "CAL", "method": c["method"], "family_ordinal": c["family_ordinal"],
                      "lr": c["lr"], "m1_cap": c["m1_cap"], "seed": c["tuning_seed"], "steps": c["steps"],
                      "checkpoint_every": c["checkpoint_every"], "cal_registrations": cost["cal_registrations"],
                      "projection_gpu_h": cost["tuning_gpu_h"], "status": "WAIT_DRIVER_AND_FREEZE"})
    for method in p["bf_batch1"]["methods"]:
        for seed in p["bf_batch1"]["seeds"]["formal"]:
            queue.append({"unit_id": f"BF1_{method}_s{seed}", "block": "FINAL_DEV", "method": method, "family_ordinal": "NA",
                          "lr": "CAL_SELECTED", "m1_cap": "CAL_SELECTED" if method == "MF_VCS_M1" else "NA", "seed": seed,
                          "steps": 3000, "checkpoint_every": 250, "cal_registrations": 960, "projection_gpu_h": "CAL_DEPENDENT",
                          "status": "WAIT_CAL_SELECTION"})
    path = OUT / "BF1_NEXT_UNITS.csv"
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(queue[0]))
        w.writeheader()
        w.writerows(queue)
    heads = {}
    for name in ("ssl_pilot_v10", "ssl_pilot_v10_bf1", "ssl_pilot_v10_t03"):
        heads[name] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / name, text=True).strip()
    shared = ("v10/T01/TORCH_VALIDATION.json", "v10/T01/GEOMETRY_GRADIENT_CHECK.json", "v10/T02/BUDGET.json",
              "v10/T02/T02_M1_RELATIVE_LONG_cuda.csv", "v10/T02/T02_M1_RELATIVE_TIMING_cuda.json")
    bf1_hash = tracked_hash(ROOT / "ssl_pilot_v10_bf1", ["src/v10", "scripts/v10"])
    synced = bf1_hash == audit["identities"]["bf_code_hash"] and all(
        sha(ROOT / "ssl_pilot_v10_bf1" / rel) == sha(ROOT / "ssl_pilot_v10" / rel) for rel in shared)
    state = {"schema": "cs_qmi.v10.takeover.v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
             "status": "T03_ARTIFACTS_AUDITED_BF1_PREPARATION_PENDING", "ready_to_submit_bf1": False,
             "heads": heads, "accepted_identities": audit["identities"], "bf1_evidence_synced": synced, "bf1_core_hash": bf1_hash,
             "sources": [{"path": str(x.relative_to(ROOT)), "sha256": sha(x)} for x in (pp, bp, dp)],
             "queue": {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "CAL": 18, "FINAL_DEV": 12},
             "budget": {"status": "MEASURED_PROJECTION", "tuning_gpu_h": b["projection"]["tuning_total_gpu_h"],
                        "tuning_and_final_gpu_h_by_M1_cap": b["projection"]["total_gpu_h_by_selected_M1_cap_max_over_lr"],
                        "actual_BF1_spending": "not established by this audit; no BF1 result artifact located"},
             "closed_evidence": ["T00 role/exposure decision recorded", "T01 required records PASS", "relative-M1 T02 measurement accepted",
                                 "T03 smoke 7, CAL 252, formal 960 artifacts and descriptive aggregation audited"],
             "next_required_work": [
                 {"id": "BF1-01", "action": "Bring accepted 4336bb6 validation/timing records into BF1 worktree and confirm unchanged core hash.", "state": "COMPLETE" if synced else "PENDING"},
                 {"id": "BF1-02", "action": "Implement full train/checkpoint/CAL/DEV driver, append-only actual family cost ledger and safe resume checks.", "state": "PENDING"},
                 {"id": "BF1-03", "action": "Freeze the eight common protocol items and accepted immutable driver snapshot; smoke the real path under Slurm.", "state": "PENDING"},
                 {"id": "BF1-04", "action": "Run 18 CAL candidates in frozen within-family order; select with the recorded lexicographic rule.", "state": "WAIT_PREDECESSORS"},
                 {"id": "BF1-05", "action": "Run the 12 final training units and DEV32 readouts, including DEV24 sensitivity and core/group uncertainty.", "state": "WAIT_CAL"},
                 {"id": "BF1-06", "action": "Complete required remaining baseline adapters, targeted gates, timing and attachments before comparative claims.", "state": "PENDING"}],
             "held_data": {"TEST134": "HOLD; later TEST_REPEAT only after complete V10 freeze and a new manifest", "CONFIRM": 0, "EPFL": "NOT_ACTIVATED"},
             "cluster": read(OUT / "evidence/SLURM_ACCESS.json"),
             "policy": "Do not use completed T03 EVAL/truth to revise BF1 choices; do not silently reuse COMPLETE with different identity/configuration."}
    (OUT / "V10_TAKEOVER_STATE.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"CAL": 18, "FINAL_DEV": 12, "ready_to_submit_bf1": False, "tuning_projection_gpu_h": state["budget"]["tuning_gpu_h"]}))


if __name__ == "__main__":
    main()
