#!/usr/bin/env python3
"""Lightweight checks of handoff links, evidence bindings and audit guard helpers."""
import ast
import hashlib
import json
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote

from audit_v10 import Audit, best_candidate, best_checkpoint, basic_stats

ROOT = Path(__file__).resolve().parents[2]


def main():
    checks = []
    for name in ("PROJECT_MEMORY_CN.md", "CODE_REVIEW_CN.md", "V10_TAKEOVER_CN.md", "handoff/evaluator_only/V10_T03_READOUT_CN.md"):
        p = ROOT / name
        for angle, ordinary in re.findall(r"\]\((?:<([^>]+)>|([^\s)]+))\)", p.read_text()):
            link = angle or ordinary
            if re.match(r"\w+://", link) or link.startswith("#"):
                continue
            target = unquote(link.split("#", 1)[0])
            assert (p.parent / target).exists(), (name, target)
            checks.append("link")
    for p in (ROOT / "handoff/tools").glob("*.py"):
        ast.parse(p.read_text(), filename=str(p))
        checks.append("syntax")
    audit = json.loads((ROOT / "handoff/evidence/V10_AUDIT.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["audit_tool_sha256"] == hashlib.sha256((ROOT / "handoff/tools/audit_v10.py").read_bytes()).hexdigest()
    state = json.loads((ROOT / "handoff/V10_TAKEOVER_STATE.json").read_text())
    assert state["bf1_evidence_synced"] and not state["ready_to_submit_bf1"]
    for entry in state["sources"] + [state["queue"]]:
        assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
        checks.append("evidence_binding")
    with tempfile.TemporaryDirectory(prefix="cs_qmi_handoff_") as d:
        p = Path(d) / "artifact"
        p.write_bytes(b"original")
        expected = {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        good = Audit()
        good.file(p, expected, "fixture")
        assert not good.errors
        p.write_bytes(b"modified")
        bad = Audit()
        bad.file(p, expected, "fixture")
        assert any("sha256" in x["context"] for x in bad.errors)
        missing = Audit()
        missing.file(Path(d) / "missing", expected, "fixture")
        assert missing.errors
    checks += ["artifact_matches", "artifact_tamper_rejected", "missing_artifact_rejected"]
    trace = [{"step": 500, "CAL_UJ": -0.1}, {"step": 250, "CAL_UJ": -0.1}, {"step": 750, "CAL_UJ": -0.2}]
    assert best_checkpoint(trace)["step"] == 250
    records = {"later": {"config": {"lr": 0.1, "cap": None}, "checkpoints": [trace[0]]},
               "early": {"config": {"lr": 0.2, "cap": None}, "checkpoints": [trace[1]]}}
    assert best_candidate(["later", "early"], records)["unit_id"] == "early"
    records["later"]["checkpoints"] = [trace[1]]
    assert best_candidate(["later", "early"], records)["unit_id"] == "later"
    s = basic_stats([-1.0, 1.0])
    assert s["mean"] == 0 and s["rmse"] == 1 and s["var"] == 2
    checks += ["negative_CAL_kept", "checkpoint_tie_order", "candidate_tie_order", "descriptive_stats"]
    out = {"status": "PASS", "n_checks": len(checks), "scope": "Handoff artifacts and utility guard checks only; no scientific training or torch tests.", "checks": checks}
    (ROOT / "handoff/evidence/HANDOFF_CHECKS.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "n_checks": len(checks)}))


if __name__ == "__main__":
    main()
