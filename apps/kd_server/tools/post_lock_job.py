#!/usr/bin/env python
"""After `vcs_common.experiment train`: classify the receipt and, for COMPLETED, apply the storage rule.

COMPLETED: re-verify with the package's own `_completed_record` (receipt, artifact and result hashes), record
           the size + SHA256 of every run file, then delete last.pt (the resume checkpoint; the global entry
           point cannot resume and no lock/freeze/evaluate step reads it). best.pt, CAL predictions, metrics,
           history, run.json and resolved_config.json are kept.  --keep-last disables the deletion.
FAILED:    keep everything; print the amendment reminder (EXPERIMENT_LOCK §6).
RUNNING:   the training process ended without updating its receipt (killed). Keep everything; print the exact
           `record-interruption` command for the operator. Nothing is recorded automatically.
NO_RECEIPT: `train` refused before writing a receipt (e.g. lock check failed); the job ID is still unused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--train-rc", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--keep-last", action="store_true")
    args = p.parse_args()
    from vcs_common import experiment as ex
    lock = ex._load_lock(args.lock)
    job = ex._job(lock, args.job_id)
    receipt_path = ex._receipt_path(args.lock, job)
    run_dir = Path(job["run_dir"])
    result = {"schema": "kd_server_post_v1", "job_id": args.job_id, "train_rc": args.train_rc, "run_dir": str(run_dir)}
    if not receipt_path.exists():
        result["status"] = "NO_RECEIPT"
    else:
        receipt = json.loads(receipt_path.read_text())
        result["status"] = receipt.get("status")
        result["receipt_sha256"] = sha256(receipt_path)
        if receipt.get("status") == "FAILED":
            result["error"] = {"type": receipt.get("error_type"), "message": receipt.get("error")}
            result["next"] = "Candidate FAILED: select-cal/seal are blocked. Keep all files; a pre-TEST amendment with a fresh grid is required (EXPERIMENT_LOCK §6)."
        elif receipt.get("status") == "RUNNING":
            result["next"] = ("Receipt left RUNNING. After confirming the process has stopped: "
                              f"{sys.executable} -m vcs_common.experiment record-interruption --lock {Path(args.lock).resolve()} "
                              f"--job-id {args.job_id} --reason \"<confirmed fact>\"; then amend (EXPERIMENT_LOCK §6).")
    if run_dir.is_dir():
        result["files"] = {q.name: {"bytes": q.stat().st_size} for q in sorted(run_dir.iterdir()) if q.is_file()}
    if result["status"] == "COMPLETED":
        record = ex._completed_record(args.lock, lock, job)  # raises if anything bound by the receipt changed
        result["completed_record"] = {k: record[k] for k in ("result_sha256", "checkpoint_sha256", "cal_predictions_sha256", "metrics_sha256", "receipt_sha256")}
        for name, meta in result.get("files", {}).items():
            meta["sha256"] = sha256(run_dir / name)
        last = run_dir / "last.pt"
        if last.is_file() and not args.keep_last:
            result["last_pt_deleted"] = {"bytes": last.stat().st_size, "sha256": sha256(last)}
            last.unlink()
            result["storage_rule"] = "resume checkpoint kept only while running; CAL-selected best.pt + predictions kept"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result.get(k) for k in ("status", "train_rc", "next")}), flush=True)


if __name__ == "__main__":
    main()
