#!/usr/bin/env python
"""Strict FIT teacher helpers for kd_teacher.sbatch (the teacher is a native `kd train-teacher` run, not a
locked candidate, so the package's epoch-boundary --resume is usable).

mode <out>:  prints one of  complete | resume | fresh
   complete  run.json status complete and teacher.pt + provenance present (nothing to do)
   resume    run.json not complete, last.pt present and history.json has exactly last.pt's epoch + 1 entries
   fresh     no run yet; or an unusable partial run (no last.pt, or history/checkpoint disagree, which would
             lose an epoch record on resume) - that directory is renamed aside, never deleted
post <out> <rc>: verifies the finished teacher against the role manifest (FIT role, split hash, checkpoint
   hash, fixed final epoch, no TEST selection), makes teacher.pt/provenance read-only, records then deletes
   last.pt (storage rule; teacher.pt holds the final-epoch weights the students load).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def move_aside(out, reason):
    target = out.with_name(f"{out.name}.attempt_{stamp()}_{reason}")
    out.rename(target)
    print(f"moved unusable partial run to {target}", file=sys.stderr)


def mode(out):
    if not (out / "run.json").is_file():
        if out.exists() and any(out.iterdir()):
            move_aside(out, "no_run_json")
        return "fresh"
    run = json.loads((out / "run.json").read_text())
    if run.get("status") == "complete":
        if (out / "teacher.pt").is_file() and (out / "teacher_provenance.json").is_file():
            return "complete"
        raise SystemExit("run.json complete but teacher.pt/provenance missing: inspect manually")
    if not (out / "last.pt").is_file():
        move_aside(out, "no_epoch_checkpoint")
        return "fresh"
    import torch
    epoch = int(torch.load(out / "last.pt", map_location="cpu", weights_only=True)["epoch"])
    history = json.loads((out / "history.json").read_text()) if (out / "history.json").is_file() else []
    if len(history) == epoch + 1:
        return "resume"
    move_aside(out, f"history{len(history)}_vs_ckpt_epoch{epoch}")
    return "fresh"


def post(out, rc, record_path):
    result = {"schema": "kd_server_teacher_post_v1", "out": str(out), "rc": rc}
    ok = rc == 0
    if ok:
        run = json.loads((out / "run.json").read_text())
        prov = json.loads((out / "teacher_provenance.json").read_text())
        cfg = run["config"]
        manifest = json.loads(Path(cfg["data"]["manifest"]).read_text())
        checks = {
            "status_complete": run.get("status") == "complete",
            "teacher_training": run.get("teacher_training") is True,
            "epochs_complete": run.get("epochs_completed") == cfg["training"]["epochs"],
            "role_FIT": prov.get("training_role") == "FIT",
            "split_matches_manifest": prov.get("split_sha256") == manifest["split_sha256"] == run.get("split_sha256"),
            "checkpoint_hash": prov.get("checkpoint_sha256") == sha256(out / "teacher.pt"),
            "fixed_final_epoch": prov.get("selection_rule") == "fixed_final_epoch" and prov.get("epoch") == cfg["training"]["epochs"] - 1,
            "no_test_selection": prov.get("test_used_for_selection") is False and run.get("test_loaded_during_train") is False,
            "fit_n_45000": prov.get("fit_n") == 45000,
            "architecture": prov.get("architecture") == cfg["teacher"]["name"],
        }
        result["checks"] = checks
        ok = all(checks.values())
        result["teacher_sha256"] = prov.get("checkpoint_sha256")
        result["provenance_sha256"] = sha256(out / "teacher_provenance.json")
        result["environment"] = run.get("environment")
        result["fit_wall_seconds_sum"] = run.get("fit_wall_seconds_sum")
        result["cuda_peak_allocated_bytes"] = run.get("cuda_peak_allocated_bytes")
        result["files"] = {q.name: q.stat().st_size for q in sorted(out.iterdir()) if q.is_file()}
        if ok:
            for name in ("teacher.pt", "teacher_provenance.json"):
                os.chmod(out / name, 0o444)
            last = out / "last.pt"
            if last.is_file():
                result["last_pt_deleted"] = {"bytes": last.stat().st_size, "sha256": sha256(last)}
                last.unlink()
    result["status"] = "PASS" if ok else "FAIL"
    Path(record_path).parent.mkdir(parents=True, exist_ok=True)
    Path(record_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result.get(k) for k in ("status", "checks", "teacher_sha256")}), flush=True)
    return 0 if ok else 1


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "mode":
        print(mode(Path(sys.argv[2])))
    elif len(sys.argv) >= 5 and sys.argv[1] == "post":
        raise SystemExit(post(Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]))
    else:
        raise SystemExit("usage: teacher_tool.py mode <out> | post <out> <rc> <record.json>")


if __name__ == "__main__":
    main()
